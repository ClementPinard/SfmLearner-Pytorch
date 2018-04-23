import numpy as np
import json
from path import Path
from scipy.misc import imread
from tqdm import tqdm


class test_framework_stillbox(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=80, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.gt_files, self.img_files, self.displacements = read_scene_data(self.root, test_files, seq_length, step)

    def __getitem__(self, i):
        tgt = imread(self.img_files[i][0]).astype(np.float32)
        depth = np.load(self.gt_files[i])
        return {'tgt': tgt,
                'ref': [imread(img).astype(np.float32) for img in self.img_files[i][1]],
                'path':self.img_files[i][0],
                'gt_depth': depth,
                'displacement': np.array(self.displacements[i]),
                'mask': generate_mask(depth, self.min_depth, self.max_depth)
                }

    def __len__(self):
        return len(self.img_files)


def get_displacements(scene, middle_index, ref_indices):
    assert(all(i < scene['length'] and i >= 0 for i in ref_indices)), str(ref_indices)
    atomic_movement = np.linalg.norm(scene['speed'])*scene['time_step']
    cum_speed = 0
    """in Still box, movements are rectilinear so magnitude adds up.
    I mean, this is very convenient, I wonder who is the genius who came with such a dataset"""
    for i,index in enumerate(ref_indices):
        if index != middle_index:
            cum_speed += atomic_movement * abs(index - middle_index)

    return cum_speed/max(len(ref_indices) - 1, 1)


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    metadata_files = {}
    for folder in data_root.dirs():
        with open(folder/'metadata.json', 'r') as f:
            metadata_files[str(folder.name)] = json.load(f)
    gt_files = []
    im_files = []
    displacements = []
    demi_length = (seq_length - 1) // 2
    shift_range = step * np.arange(-demi_length, demi_length + 1)

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        folder, file = sample.split('/')
        _, scene_index, index = file[:-4].split('_')  # filename is in the form 'RGB_XXXX_XX.jpg'
        index = int(index)
        scene = metadata_files[folder]['scenes'][int(scene_index)]
        tgt_img_path = data_root/sample
        folder_path = data_root/folder
        if tgt_img_path.isfile():
            ref_indices = shift_range + np.clip(index, step*demi_length, scene['length'] - step * demi_length - 1)
            ref_imgs_path = [folder_path/'{}'.format(scene['imgs'][ref_index]) for ref_index in ref_indices]

            gt_files.append(folder_path/'{}'.format(scene['depth'][index]))
            im_files.append([tgt_img_path,ref_imgs_path])
            displacements.append(get_displacements(scene, demi_length, ref_indices))
        else:
            print('{} missing'.format(tgt_img_path))

    return gt_files, im_files, displacements


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop gt to exclude border values
    # if used on gt_size 100x100 produces a crop of [-95, -5, 5, 95]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.05 * gt_height, 0.95 * gt_height,
                     0.05 * gt_width,  0.95 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
