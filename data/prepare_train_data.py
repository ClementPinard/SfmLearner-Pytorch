import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dataset-format", type=str, required=True, choices=["kitti", "cityscapes"])
parser.add_argument("--static-frames", default=None,
                    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)")
parser.add_argument("--with-depth", action='store_true',
                    help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--dump-root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()


def dump_example(scene):
    scene_list = data_loader.collect_scenes(scene)
    for scene_data in scene_list:
        dump_dir = args.dump_root/scene_data['rel_path']
        dump_dir.makedirs_p()
        intrinsics = scene_data['intrinsics']

        dump_cam_file = dump_dir/'cam.txt'

        np.savetxt(dump_cam_file, intrinsics)
        poses_file = dump_dir/'poses.txt'
        poses = []

        for sample in data_loader.get_scene_imgs(scene_data):
            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
            scipy.misc.imsave(dump_img_file, img)
            if "pose" in sample.keys():
                poses.append(sample["pose"].tolist())
            if "depth" in sample.keys():
                dump_depth_file = dump_dir/'{}.npy'.format(frame_nb)
                np.save(dump_depth_file, sample["depth"])
        if len(poses) != 0:
            np.savetxt(poses_file, np.array(poses).reshape(-1, 12))

        if len(dump_dir.files('*.jpg')) < 3:
            dump_dir.rmtree()


def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader

    if args.dataset_format == 'kitti':
        from kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(args.dataset_dir,
                                     static_frames_file=args.static_frames,
                                     img_height=args.height,
                                     img_width=args.width,
                                     get_depth=args.with_depth,
                                     get_pose=args.with_pose)

    if args.dataset_format == 'cityscapes':
        from cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.height,
                                        img_width=args.width)

    print('Retrieving frames')
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(scene)
    else:
        Parallel(n_jobs=args.num_threads)(delayed(dump_example)(scene) for scene in tqdm(data_loader.scenes))

    print('Generating train val lists')
    np.random.seed(8964)
    subfolders = args.dump_root.dirs()
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for s in tqdm(subfolders):
                if np.random.random() < 0.1:
                    vf.write('{}\n'.format(s.name))
                else:
                    tf.write('{}\n'.format(s.name))

                    if args.with_gt:
                        # remove useless groundtruth data for training comment if you don't want to erase it
                        s/'poses.txt'.remove_p()
                        for gt_file in s.files('*.npy'):
                            gt_file.remove_p()


if __name__ == '__main__':
    main()
