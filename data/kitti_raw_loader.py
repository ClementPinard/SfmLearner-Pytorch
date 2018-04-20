from __future__ import division
import numpy as np
from path import Path
import scipy.misc
from collections import Counter


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def pose_from_oxts_packet(metadata, scale):

    lat, lon, alt, roll, pitch, yaw = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 static_frames_file=None,
                 img_height=128,
                 img_width=416,
                 min_speed=2,
                 get_depth=False,
                 get_pose=False,
                 depth_size_ratio=1):
        dir_path = Path(__file__).realpath().dirname()
        test_scene_file = dir_path/'test_scenes.txt'

        self.from_speed = static_frames_file is None
        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)

        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.min_speed = min_speed
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
            scene_data = {'cid': c, 'dir': drive, 'speed': [], 'frame_id': [], 'pose':[], 'rel_path': drive.name + '_' + c}
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
            scene_data['intrinsics'] = scene_data['P_rect'][:,:3]

            train_scenes.append(scene_data)
        return train_scenes

    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"img":self.load_image(scene_data, i)[0], "id":frame_id}

            if self.get_depth:
                sample['depth'] = self.generate_depth_map(scene_data, i)
            if self.get_pose:
                sample['pose'] = scene_data['pose'][i]
            return sample

        if self.from_speed:
            cum_speed = np.zeros(3)
            for i, speed in enumerate(scene_data['speed']):
                cum_speed += speed
                speed_mag = np.linalg.norm(cum_speed)
                if speed_mag > self.min_speed:
                    frame_id = scene_data['frame_id'][i]
                    yield construct_sample(scene_data, i, frame_id)
                    cum_speed *= 0
        else:  # from static frame file
            drive = str(scene_data['dir'].name)
            for (i,frame_id) in enumerate(scene_data['frame_id']):
                if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                    yield construct_sample(scene_data, i, frame_id)

    def get_P_rect(self, scene_data, zoom_x, zoom_y):
        calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
        P_rect[0] *= zoom_x
        P_rect[1] *= zoom_y
        return P_rect

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

    def generate_depth_map(self, scene_data, tgt_idx):
        # compute projection matrix velodyne->image plane

        def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n-1) + colSub - 1

        R_cam2rect = np.eye(4)

        calib_dir = scene_data['dir'].parent
        cam2cam = self.read_raw_calib_file(calib_dir/'calib_cam_to_cam.txt')
        velo2cam = self.read_raw_calib_file(calib_dir/'calib_velo_to_cam.txt')
        velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        P_rect = np.copy(scene_data['P_rect'])
        P_rect[0] /= self.depth_size_ratio
        P_rect[1] /= self.depth_size_ratio

        R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)

        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        velo_file_name = scene_data['dir']/'velodyne_points'/'data'/'{}.bin'.format(scene_data['frame_id'][tgt_idx])

        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = np.fromfile(velo_file_name, dtype=np.float32).reshape(-1, 4)
        velo[:,3] = 1
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1

        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:,0] < self.img_width/self.depth_size_ratio)
        val_inds = val_inds & (velo_pts_im[:,1] < self.img_height/self.depth_size_ratio)
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros((self.img_height // self.depth_size_ratio, self.img_width // self.depth_size_ratio)).astype(np.float32)
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0
        return depth
