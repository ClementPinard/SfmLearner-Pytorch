import numpy as np
from .sequence_folders import SequenceFolder, load_as_float
import random
import json


class ShiftedSequenceFolder(SequenceFolder):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        (optional) root/scene_1/shifts.json
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, target_displacement=0.02, transform=None, target_transform=None):
        super().__init__(root, seed, train, sequence_length, transform, target_transform)
        self.target_displacement = target_displacement
        self.max_shift = 10
        self.adjust = False

    def crawl_folders(self, sequence_length):
        sequence_set = []
        img_sequences = []
        demi_length = (sequence_length-1)//2
        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue

            shifts_file = scene/'shifts.json'
            if shifts_file.isfile():
                with open(shifts_file, 'r') as f:
                    shifts = json.load(f)
            else:
                shifts = list(range(-demi_length, demi_length + 1))
                shifts.pop(demi_length)
                shifts = [shifts[:] for i in imgs]

            img_sequences.append(imgs)
            sequence_index = len(img_sequences) - 1
            intrinsics = np.genfromtxt(scene/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': i, 'ref_imgs': shifts[i], 'sequence_index': sequence_index}
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        self.img_sequences = img_sequences

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = self.img_sequences[sample['sequence_index']]
        tgt_index = sample['tgt']
        tgt_img = load_as_float(imgs[tgt_index])
        try:
            ref_imgs = [load_as_float(imgs[tgt_index + i]) for i in sample['ref_imgs']]
        except Exception as e:
            print(index, sample['tgt'], sample['ref_imgs'], len(imgs))
            raise e
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, sample['intrinsics'])
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = sample['intrinsics']
        sample = (tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics))

        if self.adjust:
            return (index, *sample)
        else:
            return sample

    def reset_shifts(self, index, displacements):
        sample = self.samples[index]
        assert(len(sample['ref_imgs']) == len(displacements))
        imgs = self.img_sequences[sample['sequence_index']]
        tgt = sample['tgt']
        mid_index = int(len(displacements)/2)
        # split the shift list in two by sign and ascending magnitude
        # e.g. [-2,-1,1,2] becomes [-1,-2], [1,2]
        before = displacements[mid_index-1::-1]

        after = displacements[mid_index:]

        for j, d in enumerate(before):
            target = (j+1)*self.target_displacement
            shift_index = mid_index - j - 1
            new_shift = sample['ref_imgs'][shift_index] * target/d
            assert(new_shift < 0)
            new_shift = round(new_shift)
            ''' Here is how bounds work for anterior shifts:
            anterior shifts must be negative in a strict ascending order in the original list
            max_shift (in magnitude) is either tgt (to keep index inside list) or self.max_shift
            Let's say you have 2 anterior shifts, which means seq_length is 5
            1st shift can be -max_shift but cannot be 0 as it would mean that 2nd would not be higher than 1st and above 0
            2nd shift cannot be -max_shift as 1st shift would have to be less than -max_shift - 1.
            More generally, shift must be clipped within -max_shift + its index and upper shift - 1
            '''

            max_shift = min(tgt, self.max_shift)

            lower_bound = -max_shift + shift_index
            upper_bound = -1 if j == 0 else sample['ref_imgs'][shift_index + 1] - 1

            sample['ref_imgs'][mid_index-j-1] = int(np.clip(new_shift, lower_bound, upper_bound))

        for j, d in enumerate(after):
            target = (j+1)*self.target_displacement
            shift_index = mid_index + j
            new_shift = sample['ref_imgs'][shift_index] * target/d
            assert(new_shift > 0)
            new_shift = round(new_shift)
            '''For posterior shifts :
            must be postive in a strict descending order
            max_shift is either len(imgs) - tgt or self.max_shift
            shift must be clipped within upper shift + 1 and max_shift - seq_length + its index
            '''

            max_shift = min(len(imgs) - tgt - 1, self.max_shift)

            lower_bound = 1 if j == 0 else sample['ref_imgs'][shift_index - 1] + 1
            upper_bound = max_shift - len(displacements) + shift_index

            sample['ref_imgs'][shift_index] = int(np.clip(new_shift, lower_bound, upper_bound))

    def __len__(self):
        return len(self.samples)
