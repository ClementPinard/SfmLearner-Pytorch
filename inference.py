from path import Path
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import torch
from torch.nn import functional as F
from models import DispNetS, PoseExpNet
from skimage.transform import resize
from evaluation_toolkit.inference_toolkit import inferenceFramework


@torch.no_grad()
def main():
    parser = ArgumentParser(description='Example usage of Inference toolkit',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_root', metavar='DIR', type=Path, required=True)
    parser.add_argument('--depth_output', metavar='FILE', type=Path, required=True,
                        help='where to store the estimated depth maps, must be a npy file')
    parser.add_argument('--evaluation_list_path', metavar='PATH', type=Path, required=True,
                        help='File with list of images to test for depth evaluation')
    parser.add_argument('--pretrained_dispnet', metavar='FILE', type=Path, required=True)
    parser.add_argument('--pretrained_posenet', metavar='FILE', default=None, type=Path)
    parser.add_argument('--no-resize', action='store_true')
    parser.add_argument("--img-height", default=128, type=int, help="Image height")
    parser.add_argument("--img-width", default=416, type=int, help="Image width")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(args.evaluation_list_path) as f:
        evaluation_list = [line[:-1] for line in f.readlines()]

    def preprocessing(frame):
        h, w, _ = frame.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            frame = resize(frame, (args.img_height, args.img_width))
        frame_np = (frame.transpose(2, 0, 1).astype(np.float32)[None]/255 - 0.5)/0.5
        return torch.from_numpy(frame_np).to(device)

    engine = inferenceFramework(args.dataset_root, evaluation_list, frame_transform=preprocessing)

    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet is not None:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    for sample in tqdm(engine):
        tgt_img, latest_intrinsics, poses = sample.get_frame()

        pred_disp = disp_net(tgt_img)
        pred_depth = 1/pred_disp
        if (not args.no_resize) and (pred_disp.shape[0] != args.img_height or pred_disp.shape[1] != args.img_width):
            pred_depth = F.interpolate(pred_depth, size=(args.img_height, args.img_width), align_corners=True)
        pred_depth = pred_depth.cpu().numpy()[0, 0]

        if args.pretrained_posenet is not None:
            shifts = range(1, seq_length)

            ref_imgs, previous_intrinsics, previous_poses = sample.get_previous_frames(shifts)
            ref_imgs = list(ref_imgs)[::-1] + [tgt_img]
            # Reorganize ref_imgs : tgt is middle frame but not necessarily the one used in DispNetS
            # (in case sample to test was in end or beginning of the image sequence)
            middle_index = seq_length//2
            tgt = ref_imgs[middle_index]
            reorganized_refs = ref_imgs[:middle_index] + ref_imgs[middle_index + 1:]
            _, poses = pose_net(tgt, reorganized_refs)
            estimated_displacement_magnitudes = poses[0, :, :3].norm(2, 1).cpu().numpy()
            GT_displacement_magnitudes = np.linalg.norm(np.stack(previous_poses)[:, :, -1], axis=-1)

            scale_factor = np.mean(GT_displacement_magnitudes / estimated_displacement_magnitudes)
            pred_depth *= scale_factor

        engine.finish_frame(pred_depth)
    mean_inference_time, output_depth_maps = engine.finalize(output_path=args.depth_output)

    print("Mean time per sample : {:.2f}us".format(1e6 * mean_inference_time))
    np.savez(args.depth_output, **output_depth_maps)


if __name__ == '__main__':
    main()
