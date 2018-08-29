import time
import csv

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import models
from utils import save_checkpoint,save_path_formatter
from logger import TermLogger, AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter
from datasets.shifted_sequence_folders import ShiftedSequenceFolder
from datasets.sequence_folders import SequenceFolder
from train import train, validate_with_gt, validate_without_gt, parser

parser.add_argument('-d', '--target-displacement', type=float, help='displacement to aim at when adjustting shifts, regarding posenet output',
                    metavar='D', default=0.05)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global args, best_error, n_iter, device
    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints_shifted'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = ShiftedSequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        target_displacement=args.target_displacement
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    adjust_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)  # workers is set to 0 to avoid multiple instances to be modified at the same time
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    train.args = args
    # create model
    print("=> creating model")

    disp_net = models.DispNetS().cuda()
    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).to(device)

    if args.pretrained_exp_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> setting adam solver')

    parameters = chain(disp_net.parameters(), pose_exp_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        if (epoch + 1) % 5 == 0:
            train_set.adjust = True
            logger.reset_train_bar(len(adjust_loader))
            average_shifts = adjust_shifts(args, train_set, adjust_loader, pose_exp_net, epoch, logger, training_writer)
            shifts_string = ' '.join(['{:.3f}'.format(s) for s in average_shifts])
            logger.train_writer.write(' * adjusted shifts, average shifts are now : {}'.format(shifts_string))
            for i, shift in enumerate(average_shifts):
                training_writer.add_scalar('shifts{}'.format(i), shift, epoch)
            train_set.adjust = False

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


@torch.no_grad()
def adjust_shifts(args, train_set, adjust_loader, pose_exp_net, epoch, logger, train_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    new_shifts = AverageMeter(args.sequence_length-1)
    pose_exp_net.train()
    poses = np.zeros(((len(adjust_loader)-1) * args.batch_size * (args.sequence_length-1),6))

    end = time.time()

    for i, (indices, tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(adjust_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]

        # compute output
        explainability_mask, pose_batch = pose_exp_net(tgt_img, ref_imgs)

        if i < len(adjust_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose_batch.cpu().reshape(-1,6).numpy()

        for index, pose in zip(indices, pose_batch):
            displacements = pose[:,:3].norm(p=2, dim=1).cpu().numpy()
            train_set.reset_shifts(index, displacements)
            new_shifts.update(train_set.samples[index]['ref_imgs'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i)
        if i % args.print_freq == 0:
            logger.train_writer.write('Adjustement:'
                                      'Time {} Data {} shifts {}'.format(batch_time, data_time, new_shifts))

    prefix = 'train poses'
    coeffs_names = ['tx', 'ty', 'tz']
    if args.rotation_mode == 'euler':
        coeffs_names.extend(['rx', 'ry', 'rz'])
    elif args.rotation_mode == 'quat':
        coeffs_names.extend(['qx', 'qy', 'qz'])
    for i in range(poses.shape[1]):
        train_writer.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses[:,i], epoch)

    return new_shifts.avg


if __name__ == '__main__':
    main()
