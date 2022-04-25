import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
from natsort import natsorted


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        data/mp3d_sfm/split/scene_name/episode_num/
            depth/0000000.jpg
            ...
            rgb/0000000.jpg
            ...
            rgb/cam.txt
            ...
            semantics/0000000.jpg
            ..
        
        transform functions must take in a list a images and a numpy array 
        (usually intrinsics matrix)
    """

    def __init__(
        self, root, seed=None, train=True, sequence_length=3, transform=None, 
        target_transform=None):
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.root = Path(root)
        
        scene_list_path = self.root/'val.txt'
        if train:
            scene_list_path = self.root/'train.txt'
        
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.semantic_scenes = [
            Path(scene.replace('rgb', 'semantics')) for scene in self.scenes]
            
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for s, scene in enumerate(self.scenes):
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            
            # TODO: use natsorted?
            sem_imgs = natsorted(self.semantic_scenes[s].files('*.png'))
            imgs = natsorted(scene.files('*.png'))
            
            if len(imgs) < sequence_length:
                continue
            
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {
                    'intrinsics': intrinsics, 
                    'tgt': imgs[i], 
                    'tgt_sem': sem_imgs[i],
                    'ref_imgs': []
                }
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
                
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_sem_img = load_as_float(sample['tgt_sem'])
        
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            
            sem_imgs, _ = self.transform(
                [tgt_sem_img], np.copy(sample['intrinsics']))
            
            tgt_sem_img = sem_imgs[0]
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        return tgt_img, tgt_sem_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
