# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
from collections import Counter
from path import Path
from scipy.misc import imread
from tqdm import tqdm


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
                'displacement': np.array(self.displacements[i]),
                'mask': generate_mask(depth, self.min_depth, self.max_depth)
                }

    def __len__(self):
        return len(self.img_files)


###############################################################################
#  EIGEN

def getXYZ(lat, lon, alt):
    """Helper method to compute a R(3) pose vector from an OXTS packet.
    Unlike KITTI official devkit, we use sinusoidal projection (https://en.wikipedia.org/wiki/Sinusoidal_projection)
    instead of mercator as it is much simpler.
    Initially Mercator was used because it renders nicely for Odometry vizualisation, but we don't need that here.
    In order to avoid problems for potential other runs closer to the pole in the future,
    we stick to sinusoidal which keeps the distances cleaner than mercator (and that's the only thing we want here)
    See https://github.com/utiasSTARS/pykitti/issues/24
    """
    er = 6378137.  # earth radius (approx.) in meters
    scale = np.cos(lat * np.pi / 180.)
    tx = scale * lon * np.pi * er / 180.
    ty = er * lat * np.pi / 180.
    tz = alt
    t = np.array([tx, ty, tz])
    return t


def get_displacements(oxts_root, indices, tgt_index):
    """gets mean displacement magntidue between middle frame and other frames, this is, to a scaling factor
    the mean output PoseNet should have for translation. Since the scaling is the same factor for depth maps and
    for translations, it will be used to determine how much predicted depth should be multiplied to."""
    first_pose = None
    displacement = 0
    if len(indices) == 0:
        return 0
    reordered_indices = [indices[tgt_index]] + [*indices[:tgt_index]] + [*indices[tgt_index + 1:]]
    for index in reordered_indices:
        oxts_data = np.genfromtxt(oxts_root/'data'/'{:010d}.txt'.format(index))
        lat, lon, alt = oxts_data[:3]
        pose = getXYZ(lat, lon, alt)
        if first_pose is None:
            first_pose = pose
        else:
            displacement += np.linalg.norm(pose - first_pose)
    return displacement / max(len(indices - 1), 1)


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    gt_files = []
    calib_dirs = []
    im_files = []
    cams = []
    displacements = []
    demi_length = (seq_length - 1) // 2
    shift_range = step * np.arange(-demi_length, demi_length + 1)

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        tgt_img_path = data_root/sample
        date, scene, cam_id, _, index = sample[:-4].split('/')

        scene_length = len(tgt_img_path.parent.files('*.png'))

        ref_indices = shift_range + np.clip(int(index), step*demi_length, scene_length - step*demi_length - 1)

        ref_imgs_path = [tgt_img_path.dirname()/'{:010d}.png'.format(i) for i in ref_indices]
        vel_path = data_root/date/scene/'velodyne_points'/'data'/'{}.bin'.format(index[:10])

        if tgt_img_path.isfile():
            gt_files.append(vel_path)
            calib_dirs.append(data_root/date)
            im_files.append([tgt_img_path,ref_imgs_path])
            cams.append(int(cam_id[-2:]))
            displacements.append(get_displacements(data_root/date/scene/'oxts', ref_indices, demi_length))
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
