import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path, sequence_length):
    stack = imread(path).astype(np.float32)
    h,w,_ = stack.shape
    w_img = int(w/(sequence_length))
    imgs = [stack[:,i*w_img:(i+1)*w_img] for i in range(sequence_length)]
    tgt_index = sequence_length//2
    return([imgs[tgt_index]] + imgs[:tgt_index] + imgs[tgt_index+1:])


class SequenceFolder(data.Dataset):
    """A sequence data loader where the images are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000_cam.txt
        root/scene_1/0000001.jpg
        root/scene_1/0000001_cam.txt
        .
        root/scene_2/0000000.jpg
        root/scene_2/0000000_cam.txt
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.samples = []
        frames_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = self.root.dirs()
        self.sequence_length = sequence_length
        for frame_path in open(frames_list_path):
            a,b = frame_path[:-1].split(' ')
            base_path = (self.root/a)/b
            intrinsics = np.genfromtxt(base_path+'_cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            sample = {'intrinsics': intrinsics, 'img_stack': base_path+'.jpg'}
            self.samples.append(sample)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = load_as_float(sample['img_stack'], self.sequence_length)
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, np.copy(sample['intrinsics']))
        else:
            intrinsics = sample['intrinsics']
        return imgs[0], imgs[1:], intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
