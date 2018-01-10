# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
# import pandas as pd
import datetime
from collections import Counter
from path import Path
from scipy.misc import imread
from tqdm import tqdm

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


class test_framework_KITTI(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=100, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.calib_dirs, self.gt_files, self.img_files, self.displacements, self.cams = read_scene_data(self.root, test_files, seq_length, step)

    def __getitem__(self, i):
        tgt = imread(self.img_files[i][0]).astype(np.float32)
        depth = generate_depth_map(self.calib_dirs[i], self.gt_files[i], tgt.shape[:2], self.cams[i])
        return {'tgt': tgt,
                'ref': [imread(img).astype(np.float32) for img in self.img_files[i][1]],
                'path':self.img_files[i][0],
                'gt_depth': depth,
                'displacements': np.array(self.displacements[i]),
                'mask': generate_mask(depth, self.min_depth, self.max_depth)
                }

    def __len__(self):
        return len(self.img_files)


###############################################################################
#  EIGEN

def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines


def get_displacements(oxts_root, index, shifts):
    with open(oxts_root/'timestamps.txt') as f:
        timestamps = [datetime.datetime.strptime(ts[:-3], "%Y-%m-%d %H:%M:%S.%f").timestamp() for ts in f.read().splitlines()]
    oxts_data = np.genfromtxt(oxts_root/'data'/'{:010d}.txt'.format(index))
    speed = np.linalg.norm(oxts_data[8:11])
    assert(all(index+shift < len(timestamps) and index+shift >= 0 for shift in shifts)), str([index+shift for shift in shifts])
    return [speed*abs(timestamps[index] - timestamps[index + shift]) for shift in shifts]


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    gt_files = []
    calib_dirs = []
    im_files = []
    cams = []
    displacements = []
    demi_length = (seq_length - 1) // 2
    shift_range = [step*i for i in list(range(-demi_length,0)) + list(range(1, demi_length + 1))]

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        tgt_img_path = data_root/sample
        date, scene, cam_id, _, index = sample[:-4].split('/')

        ref_imgs_path = [tgt_img_path.dirname()/'{:010d}.png'.format(int(index) + shift) for shift in shift_range]

        caped_shift_range = shift_range[:]  # ensures ref_imgs are present, if not, set shift to 0 so that it will be discarded later
        for i,img in enumerate(ref_imgs_path):
            if not img.isfile():
                ref_imgs_path[i] = tgt_img_path
                caped_shift_range[i] = 0

        vel_path = data_root/date/scene/'velodyne_points'/'data'/'{}.bin'.format(index[:10])

        if tgt_img_path.isfile():
            gt_files.append(vel_path)
            calib_dirs.append(data_root/date)
            im_files.append([tgt_img_path,ref_imgs_path])
            cams.append(int(cam_id[-2:]))
            displacements.append(get_displacements(data_root/date/scene/'oxts', int(index), caped_shift_range))
        else:
            print('{} missing'.format(tgt_img_path))
    # print(num_probs, 'files missing')

    return calib_dirs, gt_files, im_files, displacements, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
    return points


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


def get_focal_length_baseline(calib_dir, cam=2):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam == 2:
        focal_length = P2_rect[0,0]
    elif cam == 3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir/'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir/'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
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


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
