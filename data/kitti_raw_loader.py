from __future__ import division
import numpy as np
from path import Path
from imageio import imread
from skimage.transform import resize as imresize
from kitti_util import pose_from_oxts_packet, generate_depth_map, read_calib_file, transform_from_rot_trans
from datetime import datetime


class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 static_frames_file=None,
                 img_height=128,
                 img_width=416,
                 min_disp=0.2,
                 get_depth=False,
                 get_pose=False,
                 depth_size_ratio=1):
        dir_path = Path(__file__).realpath().dirname()
        test_scene_file = dir_path/'test_scenes.txt'

        self.from_speed = static_frames_file is None
        if static_frames_file is not None:
            self.collect_static_frames(static_frames_file)

        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.min_disp = min_disp
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.depth_size_ratio = depth_size_ratio
        self.collect_train_folders()

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = {}
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            if drive not in self.static_frames.keys():
                self.static_frames[drive] = []
            self.static_frames[drive].append(curr_fid)

    def collect_train_folders(self):
        self.scenes = []
        for date in self.date_list:
            drive_set = (self.dataset_dir/date).dirs()
            for dr in drive_set:
                if dr.name[:-5] not in self.test_scenes:
                    self.scenes.append(dr)

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            oxts = sorted((drive/'oxts'/'data').files('*.txt'))
            with open(drive/'oxts'/'timestamps.txt', 'r') as f:
                times = [datetime.strptime(time_string[:-4], "%Y-%m-%d %H:%M:%S.%f") for time_string in f.readlines()]
            scene_data = {'cid': c,
                          'dir': drive,
                          'speed': [],
                          'time': [t.timestamp() for t in times],
                          'frame_id': [],
                          'pose': [],
                          'rel_path': drive.name + '_' + c}
            scale = None
            origin = None
            imu2velo = read_calib_file(drive.parent/'calib_imu_to_velo.txt')
            velo2cam = read_calib_file(drive.parent/'calib_velo_to_cam.txt')
            cam2cam = read_calib_file(drive.parent/'calib_cam_to_cam.txt')

            velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
            imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
            cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

            imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)
                speed = metadata[8:11]
                scene_data['speed'].append(speed)
                scene_data['frame_id'].append('{:010d}'.format(n))
                lat = metadata[0]

                if scale is None:
                    scale = np.cos(lat * np.pi / 180.)

                pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
                if origin is None:
                    origin = pose_matrix

                odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
                scene_data['pose'].append(odo_pose[:3])

            sample = self.load_image(scene_data, 0)
            if sample is None:
                return []
            scene_data['P_rect'] = self.get_P_rect(scene_data, sample[1], sample[2])
            scene_data['intrinsics'] = scene_data['P_rect'][:, :3]

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

        if self.from_speed:
            cum_displacement = np.zeros(3)
            for i, (speed1, speed2, t1, t2) in enumerate(zip(scene_data['speed'][1:],
                                                             scene_data['speed'][:-1],
                                                             scene_data['time'][1:],
                                                             scene_data['time'][:-1])):
                print(speed1, speed2, t1, t2)
                cum_displacement += 0.5*(speed1 + speed2) / (t2-t1)
                disp_mag = np.linalg.norm(cum_displacement)
                if disp_mag > self.min_disp:
                    frame_id = scene_data['frame_id'][i]
                    yield construct_sample(scene_data, i, frame_id)
                    cum_displacement *= 0
        else:  # from static frame file
            drive = str(scene_data['dir'].name)
            for (i, frame_id) in enumerate(scene_data['frame_id']):
                if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                    yield construct_sample(scene_data, i, frame_id)

    def get_P_rect(self, scene_data, zoom_x, zoom_y):
        calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        filedata = read_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
        P_rect[0] *= zoom_x
        P_rect[1] *= zoom_y
        return P_rect

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            return None
        img = imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = imresize(img, (self.img_height, self.img_width))

        # workaround for skimage (float [0 .. 1]) and imageio (uint8 [0 .. 255]) interoperability
        img = (img * 255).astype(np.uint8)

        return img, zoom_x, zoom_y

    def get_depth_map(self, scene_data, tgt_idx):
        # compute projection matrix velodyne->image plane

        R_cam2rect = np.eye(4)

        calib_dir = scene_data['dir'].parent
        cam2cam = read_calib_file(calib_dir/'calib_cam_to_cam.txt')
        velo2cam = read_calib_file(calib_dir/'calib_velo_to_cam.txt')
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

        velo2cam = np.dot(R_cam2rect, velo2cam)

        velo_file_name = scene_data['dir']/'velodyne_points'/'data'/'{}.bin'.format(scene_data['frame_id'][tgt_idx])

        return generate_depth_map(velo_file_name, scene_data['P_rect'], velo2cam,
                                  self.img_width, self.img_height, self.depth_size_ratio)
