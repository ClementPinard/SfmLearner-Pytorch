# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
# import pandas as pd
from path import Path
from scipy.misc import imread
from tqdm import tqdm


class test_framework_KITTI(object):
    def __init__(self, root, sequence_set, seq_length=3, step=1):
        self.root = root
        self.img_files, self.poses, self.sample_indices = read_scene_data(self.root, sequence_set, seq_length, step)

    def generator(self):
        for img_list, pose_list, sample_list in zip(self.img_files, self.poses, self.sample_indices):
            for snippet_indices in sample_list:
                imgs = [imread(img_list[i]).astype(np.float32) for i in snippet_indices]

                poses = np.stack(pose_list[i] for i in snippet_indices)
                first_pose = poses[0]
                poses[:,:,-1] -= first_pose[:,-1]
                compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

                yield {'imgs': imgs,
                       'path': img_list[0],
                       'poses': compensated_poses
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(imgs) for imgs in self.img_files)


def read_scene_data(data_root, sequence_set, seq_length=3, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

    sequences = set()
    for seq in sequence_set:
        corresponding_dirs = set((data_root/'sequences').dirs(seq))
        sequences = sequences | corresponding_dirs

    print('getting test metadata for theses sequences : {}'.format(sequences))
    for sequence in tqdm(sequences):
        poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence.name)).astype(np.float64).reshape(-1, 3, 4)
        imgs = sorted((sequence/'image_2').files('*.png'))
        # construct 5-snippet sequences
        tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)
    return im_sequences, poses_sequences, indices_sequences