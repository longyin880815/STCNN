from __future__ import division

import os
import numpy as np
import cv2

import pickle
import random
import imageio
import cv2
from PIL import Image
from torch.utils.data import Dataset
import skimage.morphology as sm
import random

class MSRADataset(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, inputRes=None,
                 dataset_dir='/home/xk/Dataset/MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/Imgs/',
                 transform=None
                 ):
        self.images_list = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg")])
        self.gts_list = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".png")])
        self.images_path_list = sorted([os.path.join(dataset_dir,x) for x in self.images_list])
        self.gts_path_list = sorted([os.path.join(dataset_dir, x) for x in self.gts_list])
        assert len(self.images_path_list) == len(self.gts_path_list)
        self.transform = transform
        self.inputRes = inputRes

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):

        img = imageio.imread(self.images_path_list[idx])
        gt = imageio.imread(self.gts_path_list[idx])
        if self.inputRes is not None:
            img = cv2.resize(img, (self.inputRes[1],self.inputRes[0]))
            gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]))
        img = np.array(img, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        img = img / 127.5 - 1.
        gt = gt/255.
        sample = {'images': img, 'gts': gt}
        if self.transform is not None:
            sample = self.transform(sample)
        return  sample



def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png')) for img_name in img_list]

# selems = [sm.square(5),sm.square(11),sm.square(15), sm.disk(5),sm.disk(11),sm.disk(15), sm.star(5),sm.star(11),sm.star(15)]
# selems = [sm.square(15),sm.square(25),sm.square(35), sm.disk(15),sm.disk(25),sm.disk(35), sm.star(15),sm.star(25),sm.star(35)]
selems = [cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15)),cv2.getStructuringElement(cv2.MORPH_RECT,(25, 25)),
          cv2.getStructuringElement(cv2.MORPH_RECT,(35, 35)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
          cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15)),cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25)),
          cv2.getStructuringElement(cv2.MORPH_CROSS, (35, 35))]


class ImageFolder(Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)


        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)


        sample = {'images':img, 'gts': target}
        return sample

    def __len__(self):
        return len(self.imgs)
