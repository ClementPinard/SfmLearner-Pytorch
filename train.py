import argparse
import time
import csv

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint, save_path_formatter
from inverse_warp import inverse_warp

from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
from logger import TermLogger, AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

best_error = -1
n_iter = 0


def main():
    global best_error, n_iter
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
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
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
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
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    disp_net = models.DispNetS().cuda()
    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).cuda()

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
        decisive_error = errors[1]
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


def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, logger, train_writer):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())

        # compute output
        disparities = disp_net(tgt_img_var)
        depth = [1/disp for disp in disparities]
        explainability_mask, pose = pose_exp_net(tgt_img_var, ref_imgs_var)

        loss_1 = photometric_reconstruction_loss(tgt_img_var, ref_imgs_var,
                                                 intrinsics_var, intrinsics_inv_var,
                                                 depth, explainability_mask, pose,
                                                 args.rotation_mode, args.padding_mode)
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('photometric_error', loss_1.data[0], n_iter)
            if w2 > 0:
                train_writer.add_scalar('explanability_loss', loss_2.data[0], n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_3.data[0], n_iter)
            train_writer.add_scalar('total_loss', loss.data[0], n_iter)

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)

            for k,scaled_depth in enumerate(depth):
                train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                       tensor2array(disparities[k].data[0].cpu(), max_value=None, colormap='bone'),
                                       n_iter)
                train_writer.add_image('train Depth Output Normalized {}'.format(k),
                                       tensor2array(1/disparities[k].data[0].cpu(), max_value=None),
                                       n_iter)
                b, _, h, w = scaled_depth.size()
                downscale = tgt_img_var.size(2)/h

                tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img_var, (h, w))
                ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs_var]

                intrinsics_scaled = torch.cat((intrinsics_var[:, 0:2]/downscale, intrinsics_var[:, 2:]), dim=1)
                intrinsics_scaled_inv = torch.cat((intrinsics_inv_var[:, :, 0:2]*downscale, intrinsics_inv_var[:, :, 2:]), dim=2)

                # log warped images along with explainability mask
                for j,ref in enumerate(ref_imgs_scaled):
                    ref_warped = inverse_warp(ref, scaled_depth[:,0], pose[:,j],
                                              intrinsics_scaled, intrinsics_scaled_inv,
                                              rotation_mode=args.rotation_mode,
                                              padding_mode=args.padding_mode)[0]
                    train_writer.add_image('train Warped Outputs {} {}'.format(k,j), tensor2array(ref_warped.data.cpu()), n_iter)
                    train_writer.add_image('train Diff Outputs {} {}'.format(k,j), tensor2array(0.5*(tgt_img_scaled[0] - ref_warped).abs().data.cpu()), n_iter)
                    if explainability_mask[k] is not None:
                        train_writer.add_image('train Exp mask Outputs {} {}'.format(k,j), tensor2array(explainability_mask[k][0,j].data.cpu(), max_value=1, colormap='bone'), n_iter)

        # record loss and EPE
        losses.update(loss.data[0], args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.data[0], loss_1.data[0], loss_2.data[0] if w2 > 0 else 0, loss_3.data[0]])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch, logger, output_writers=[]):
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = len(output_writers) > 0
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight
    poses = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1),6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        # compute output
        disp = disp_net(tgt_img_var)
        depth = 1/disp
        explainability_mask, pose = pose_exp_net(tgt_img_var, ref_imgs_var)

        loss_1 = photometric_reconstruction_loss(tgt_img_var, ref_imgs_var,
                                                 intrinsics_var, intrinsics_inv_var,
                                                 depth, explainability_mask, pose,
                                                 args.rotation_mode, args.padding_mode)
        loss_1 = loss_1.data[0]
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).data[0]
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth).data[0]

        if log_outputs and i % 100 == 0 and i/100 < len(output_writers):  # log first output of every 100 batch
            index = int(i//100)
            if epoch == 0:
                for j,ref in enumerate(ref_imgs):
                    output_writers[index].add_image('val Input {}'.format(j), tensor2array(tgt_img[0]), 0)
                    output_writers[index].add_image('val Input {}'.format(j), tensor2array(ref[0]), 1)

            output_writers[index].add_image('val Dispnet Output Normalized', tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'), epoch)
            output_writers[index].add_image('val Depth Output Normalized', tensor2array(1./disp.data[0].cpu(), max_value=None), epoch)
            # log warped images along with explainability mask
            for j,ref in enumerate(ref_imgs_var):
                ref_warped = inverse_warp(ref[:1], depth[:1,0], pose[:1,j],
                                          intrinsics_var[:1], intrinsics_inv_var[:1],
                                          rotation_mode=args.rotation_mode,
                                          padding_mode=args.padding_mode)[0]

                output_writers[index].add_image('val Warped Outputs {}'.format(j), tensor2array(ref_warped.data.cpu()), epoch)
                output_writers[index].add_image('val Diff Outputs {}'.format(j), tensor2array(0.5*(tgt_img_var[0] - ref_warped).abs().data.cpu()), epoch)
                if explainability_mask is not None:
                    output_writers[index].add_image('val Exp mask Outputs {}'.format(j), tensor2array(explainability_mask[0,j].data.cpu(), max_value=1, colormap='bone'), epoch)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose.data.cpu().view(-1,6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.data.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses.shape[1]):
            output_writers.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses[:,i], epoch)
        output_writers[0].add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Exp loss']


def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        depth = depth.cuda()

        # compute output
        output_disp = disp_net(tgt_img_var)
        output_depth = 1/output_disp

        if log_outputs and i % 100 == 0 and i/100 < len(output_writers):
            index = int(i//100)
            if epoch == 0:
                output_writers[index].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0].cpu()
                output_writers[index].add_image('val target Depth', tensor2array(depth_to_show, max_value=10), epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0,10)
                output_writers[index].add_image('val target Disparity Normalized', tensor2array(disp_to_show, max_value=None, colormap='bone'), epoch)

            output_writers[index].add_image('val Dispnet Output Normalized', tensor2array(output_disp.data[0].cpu(), max_value=None, colormap='bone'), epoch)
            output_writers[index].add_image('val Depth Output', tensor2array(output_depth.data[0].cpu(), max_value=3), epoch)

        errors.update(compute_errors(depth, output_depth.data))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


if __name__ == '__main__':
    main()
