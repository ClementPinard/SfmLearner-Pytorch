from __future__ import division
import numpy as np
from imageio import imread
import os
import scipy.misc
from skimage.transform import resize
from kitti_util import generate_depth_map, read_calib_file


class KittiOdomLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5,
                 min_disp=0,
                 get_depth=False,
                 get_pose=False,
                 depth_size_ratio=1):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.min_disp = min_disp
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]
        self.cams = [2, 3]
        self.scenes = self.train_seqs
        self.depth_size_ratio = depth_size_ratio

    def collect_scenes(self, scene):
        train_scenes = []
        seq_dir = self.dataset_dir / 'sequences' / '{:02d}'.format(scene)
        calib_file = seq_dir / 'calib.txt'
        calib_data = read_calib_file(calib_file)
        P0 = np.vstack([calib_data["P0"].reshape((3, 4)), [0, 0, 0, 1]])
        Tv = np.vstack([calib_data["Tr"].reshape((3, 4)), [0, 0, 0, 1]])
        for c in self.cams:
            img_dir = seq_dir / 'image_{}'.format(c)
            P_rect = calib_data["P{}".format(c)].reshape((3, 4))
            extended_P_rect = np.vstack([P_rect, [0, 0, 0, 1]])
            cam0_to_cam1 = np.linalg.inv(extended_P_rect) @ P0  # inv(T) s.t for p in R3, P.p = P0.T.p
            pose_file = self.dataset_dir / 'poses' / '{:02d}.txt'.format(scene)
            poses = np.genfromtxt(pose_file).astype(np.float64).reshape(-1, 3, 4)
            poses_4D = np.empty((poses.shape[0], 4, 4))
            poses_4D[:, :3] = poses
            poses_4D[:, -1] = [0, 0, 0, 1]
            poses_4D = cam0_to_cam1 @ poses_4D @ np.linalg.inv(cam0_to_cam1)
            poses = poses_4D[:, :3]
            imgs = sorted(img_dir.files('*.png'))
            scene_data = {'cid': c,
                          'velo2cam': Tv,
                          'dir': seq_dir,
                          'frame_path': imgs,
                          'frame_id': [i.stem for i in imgs],
                          'pose': poses,
                          'rel_path': '{:02d}_{}'.format(scene, c)}
            _, zoom_x, zoom_y = self.load_image(scene_data, 0)
            P_rect[0] *= zoom_x
            P_rect[1] *= zoom_y

            scene_data["intrinsics"] = P_rect[:, :3]
            scene_data["P_rect"] = P_rect
            train_scenes.append(scene_data)
        return train_scenes

    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"img": self.load_image(scene_data, i)[0], "id": frame_id}
            if self.get_depth:
                sample['depth'] = self.get_depth_map(scene_data, i)
            if self.get_pose:
                sample['pose'] = scene_data['pose'][i]
            return sample

        start_pose = np.zeros(3)
        for i, pose in enumerate(scene_data['pose']):
            disp_mag = np.linalg.norm(pose[:, -1] - start_pose)
            if disp_mag > self.min_disp:
                frame_id = scene_data['frame_id'][i]
                yield construct_sample(scene_data, i, frame_id)
                start_pose = pose[:, -1]

    def load_image(self, scene_data, i):
        img_file = scene_data['frame_path'][i]
        if not img_file.isfile():
            return None
        img = imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = resize(img, (self.img_height, self.img_width))

        # workaround for skimage (float [0 .. 1]) and imageio (uint8 [0 .. 255]) interoperability
        img = (img * 255).astype(np.uint8)

        return img, zoom_x, zoom_y

    def get_depth_map(self, scene_data, i):
        # compute projection matrix velodyne->image plane

        velo_file_name = scene_data['dir']/'velodyne'/'{}.bin'.format(scene_data['frame_id'][i])
        return generate_depth_map(velo_file_name, scene_data['P_rect'], scene_data['velo2cam'],
                                  self.img_width, self.img_height, self.depth_size_ratio)
