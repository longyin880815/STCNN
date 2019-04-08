from __future__ import division

import os
import numpy as np
import cv2

import pickle
import random
import imageio
import cv2
from torch.utils.data import Dataset

class VIDDataset(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, inputRes=None,
                 seqs_list_file='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                 transform=None,
                 random_rev_thred=0.4,
                 frame_len=4):

        f = open(seqs_list_file, "r")
        lines = f.readlines()
        self.seq_list = lines
        self.transform = transform
        self.inputRes = inputRes
        self.random_rev_thred = random_rev_thred
        self.frame_len = frame_len


    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):

        seq_dir = self.seq_list[idx].strip()
        frames = np.sort(os.listdir(seq_dir))
        length = len(frames)

        if random.random() > self.random_rev_thred:
            start_idx = random.randint(0, length-1-self.frame_len)
            imgs = []
            for i in range(self.frame_len):
                img = imageio.imread(os.path.join(seq_dir,frames[start_idx+i]))
                imgs.append(img)
            gt = imageio.imread(os.path.join(seq_dir,frames[start_idx+self.frame_len]))

        else:
            start_idx = random.randint(self.frame_len, length - 1)
            imgs = []
            for i in range(self.frame_len):
                img = imageio.imread(os.path.join(seq_dir, frames[start_idx - i]))
                imgs.append(img)
            gt = imageio.imread(os.path.join(seq_dir, frames[start_idx - self.frame_len]))

        imgs = np.concatenate(imgs,axis=2)
        if self.inputRes is not None:
            imgs = cv2.resize(imgs, (self.inputRes[1],self.inputRes[0]))
            gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]))
        imgs = np.array(imgs, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        sample = {'images': imgs, 'gt': gt}
        if self.transform is not None:
            sample = self.transform(sample)

        return  sample