import argparse
import numpy as np
from pebble import ProcessPool
from tqdm import tqdm
from path import Path
from imageio import imwrite

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR', type=Path,
                    help='path to original dataset')
parser.add_argument("--dataset-format", type=str, default='kitti_raw', choices=["kitti_raw", "kitti_odometry", "cityscapes"])
parser.add_argument("--static-frames", default=None, type=Path,
                    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)")
parser.add_argument("--with-depth", action='store_true',
                    help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--no-train-gt", action='store_true',
                    help="If selected, will delete ground truth depth to save space")
parser.add_argument("--dump-root", type=Path, default='dump', help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")
parser.add_argument("--depth-size-ratio", type=int, default=1, help="will divide depth size by that ratio")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()


def dump_example(args, scene):
    scene_list = data_loader.collect_scenes(scene)
    # print("scene list", scene_list, " for scene ", scene)
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
            dump_img_file.parent.makedirs_p()
            imwrite(dump_img_file, img)

            if "pose" in sample.keys():
                poses.append(sample["pose"].tolist())
            if "depth" in sample.keys():
                dump_depth_file = dump_dir/'{}.npy'.format(frame_nb)
                np.save(dump_depth_file, sample["depth"])
        if len(poses) != 0:
            np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

        if len(dump_dir.files('*.jpg')) < 3:
            dump_dir.rmtree()


def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader

    if args.dataset_format == 'kitti_raw':
        from kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(args.dataset_dir,
                                     static_frames_file=args.static_frames,
                                     img_height=args.height,
                                     img_width=args.width,
                                     get_depth=args.with_depth,
                                     get_pose=args.with_pose,
                                     depth_size_ratio=args.depth_size_ratio)

    if args.dataset_format == 'kitti_odometry':
        from kitti_odometry_loader import KittiOdomLoader
        data_loader = KittiOdomLoader(args.dataset_dir,
                                      img_height=args.height,
                                      img_width=args.width,
                                      get_depth=args.with_depth,
                                      get_pose=args.with_pose,
                                      depth_size_ratio=args.depth_size_ratio)

    if args.dataset_format == 'cityscapes':
        from cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.height,
                                        img_width=args.width)

    n_scenes = len(data_loader.scenes)
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(args, scene)
    else:
        with ProcessPool(max_workers=args.num_threads) as pool:
            tasks = pool.map(dump_example, [args]*n_scenes, data_loader.scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    print('Generating train val lists')
    np.random.seed(8964)
    # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    subdirs = args.dump_root.dirs()
    canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for pr in tqdm(canonic_prefixes):
                corresponding_dirs = args.dump_root.dirs('{}*'.format(pr))
                if np.random.random() < 0.1:
                    for s in corresponding_dirs:
                        vf.write('{}\n'.format(s.name))
                else:
                    for s in corresponding_dirs:
                        tf.write('{}\n'.format(s.name))
                        if args.with_depth and args.no_train_gt:
                            for gt_file in s.files('*.npy'):
                                gt_file.remove_p()


if __name__ == '__main__':
    main()
