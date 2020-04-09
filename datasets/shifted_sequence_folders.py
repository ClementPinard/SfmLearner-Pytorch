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
        self.max_shift = 50
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
                prior_shifts = list(range(-demi_length, 0))
                post_shifts = list(range(1, sequence_length - demi_length))
                shifts = [[prior_shifts[:], post_shifts[:]] for i in imgs]

            img_sequences.append(imgs)
            sequence_index = len(img_sequences) - 1
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics,
                          'tgt': i,
                          'prior_shifts': shifts[i][0],
                          'post_shifts': shifts[i][1],
                          'sequence_index': sequence_index}
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
            prior_imgs = [load_as_float(imgs[tgt_index + i]) for i in sample['prior_shifts']]
            post_imgs = [load_as_float(imgs[tgt_index + i]) for i in sample['post_shifts']]
            imgs = [tgt_img] + prior_imgs + post_imgs
        except Exception as e:
            print(index, sample['tgt'], sample['prior_shifts'], sample['post_shifts'], len(imgs))
            raise e
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, sample['intrinsics'])
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = sample['intrinsics']
        sample = (tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics))

        if self.adjust:
            return (index, *sample)
        else:
            return sample

    def reset_shifts(self, index, prior_ratio, post_ratio):
        sample = self.samples[index]
        assert(len(sample['prior_shifts']) == len(prior_ratio))
        assert(len(sample['post_shifts']) == len(post_ratio))
        imgs = self.img_sequences[sample['sequence_index']]
        tgt_index = sample['tgt']

        for j, r in enumerate(prior_ratio[::-1]):

            shift_index = len(prior_ratio) - 1 - j
            old_shift = sample['prior_shifts'][shift_index]
            new_shift = old_shift * r
            assert(new_shift < 0), "shift must be negative: {:.3f}, {}, {:.3f}".format(new_shift, old_shift, r)
            new_shift = round(new_shift)
            ''' Here is how bounds work for prior shifts:
            prior shifts must be negative in a strict ascending order in the original list
            max_shift (in magnitude) is either tgt (to keep index inside list) or self.max_shift
            Let's say you have 2 anterior shifts, which means seq_length is 5
            1st shift can be -max_shift but cannot be 0 as it would mean that 2nd would not be higher than 1st and above 0
            2nd shift cannot be -max_shift as 1st shift would have to be less than -max_shift - 1.
            More generally, shift must be clipped within -max_shift + its index and upper shift - 1
            Note that priority is given for shifts closer to tgt_index, they are free to choose the value they want, at the risk of
            constraining outside shifts to one only valid value
            '''

            max_shift = min(tgt_index, self.max_shift)

            lower_bound = -max_shift + shift_index
            upper_bound = -1 if shift_index == len(prior_ratio) - 1 else sample['prior_shifts'][shift_index + 1] - 1

            sample['prior_shifts'][shift_index] = int(np.clip(new_shift, lower_bound, upper_bound))

        for j, r in enumerate(post_ratio):
            shift_index = j
            old_shift = sample['post_shifts'][shift_index]
            new_shift = old_shift * r
            assert(new_shift > 0), "shift must be positive: {:.3f}, {}, {}".format(new_shift, old_shift, r)
            new_shift = round(new_shift)
            '''For posterior shifts :
            must be postive in a strict descending order
            max_shift is either len(imgs) - tgt or self.max_shift
            shift must be clipped within upper shift + 1 and max_shift - seq_length + its index
            '''

            max_shift = min(len(imgs) - tgt_index - 1, self.max_shift)

            lower_bound = 1 if shift_index == 0 else sample['post_shifts'][shift_index - 1] + 1
            upper_bound = max_shift + shift_index - len(post_ratio) + 1

            sample['post_shifts'][shift_index] = int(np.clip(new_shift, lower_bound, upper_bound))

    def get_shifts(self, index):
        sample = self.samples[index]
        prior = sample['prior_shifts']
        post = sample['post_shifts']
        return prior + post

    def __len__(self):
        return len(self.samples)
