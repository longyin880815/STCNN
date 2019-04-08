from __future__ import division

import os
import numpy as np
import cv2
import torch
import pickle
import random
import imageio
import cv2
import skimage.morphology as sm
from torch.utils.data import Dataset
from dataloaders import custom_transforms as tr


class cusToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		for elem in sample.keys():
			if 'fname' in elem:
				continue
			tmp = sample[elem]
			if isinstance(tmp,list):
				for i in range(len(tmp)):
					if tmp[i].ndim == 2:
						tmp[i] = tmp[i][:, :, np.newaxis]
					tmp[i] = tmp[i].transpose((2, 0, 1)).copy()
					tmp[i] = torch.from_numpy(tmp[i])
				sample[elem] = tmp
			else:
				if tmp.ndim == 2:
					tmp = tmp[:, :, np.newaxis]
				tmp = tmp.transpose((2, 0, 1)).copy()
				sample[elem] = torch.from_numpy(tmp)

		return sample

def im_normalize(im):
	"""
	Normalize image
	"""
	imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
	return imn

class DAVISDataset(Dataset):
	"""DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

	def __init__(self, inputRes=None,
				 samples_list_file='/home/xk/PycharmProjects/Pred_Seg/data/DAVIS16_samples_list.txt',
				 transform=None,
				 num_frame=4):

		f = open(samples_list_file, "r")
		lines = f.readlines()
		self.samples_list = lines
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):

		sample_line = self.samples_list[idx].strip()
		seq_path_line,frame_path,gt_path = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		imgs = []
		for i in range(self.num_frame):
			img = imageio.imread(seq_path[i])
			img = np.array(img, dtype=np.float32)
			imgs.append(img)

		gt = cv2.imread(gt_path, 0)
		frame = imageio.imread(frame_path)
		imgs = np.concatenate(imgs,axis=2)

		if self.inputRes is not None:
			imgs = cv2.resize(imgs, (self.inputRes[1],self.inputRes[0]))
			gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]),interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1],self.inputRes[0]))

		imgs = np.array(imgs, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)

		# normalize
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame,np.array([0.229, 0.224, 0.225], dtype=np.float32))

		sample = {'images': imgs, 'frame': frame, 'seg_gt': gt,'pred_gt': pred_gt}

		if self.transform is not None:
			sample = self.transform(sample)

		imgs = sample['images']
		imgs[np.isnan(imgs)] = 0.
		imgs[imgs > 255] = 255.0
		imgs[imgs < 0] = 0.
		imgs = imgs / 127.5 - 1.
		sample['images'] = imgs
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)
		return sample



class DAVIS_First_Frame_Dataset(Dataset):
	def __init__(self, train=True,
				 inputRes=None,
				 db_root_dir='/home/xk/Dataset/DAVIS/',
				 transform=None,
				 seq_name=None,
				 frame_nums=4):
		"""Loads image to label pairs for tool pose estimation
		db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
		"""
		self.train = train
		self.inputRes = inputRes
		self.db_root_dir = db_root_dir
		self.transform = transform
		self.seq_name = seq_name
		self.toTensor = tr.ToTensor()
		self.frame_nums = frame_nums
		# Initialize the per sequence images for online training

		names_img = np.sort([f for f in os.listdir(os.path.join(db_root_dir, 'first_frame/', str(seq_name),'dream'))
							 if f.endswith(".jpg")])
		img_list = list(map(lambda x: os.path.join(db_root_dir,'first_frame/', str(seq_name), 'dream', x), names_img))
		name_label = np.sort([f for f in os.listdir(os.path.join(db_root_dir, 'first_frame/', str(seq_name),'dream'))
							 if f.endswith(".png")])
		labels = list(map(lambda x: os.path.join(db_root_dir,'first_frame/', str(seq_name), 'dream', x), name_label))


		assert (len(labels) == len(img_list))

		self.img_list = img_list[:100]
		self.labels = labels[:100]

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
		imgs,frame, gt, pred_gt = self.make_img_gt_pair(idx)

		sample = {'images': imgs, 'frame': frame, 'seg_gt': gt, 'pred_gt': pred_gt}

		if self.seq_name is not None:
			fname = os.path.join(self.seq_name, "%05d" % idx)
			sample['fname'] = fname

		if self.transform is not None:
			sample = self.transform(sample)

		imgs = sample['images']
		imgs[np.isnan(imgs)] = 0.
		imgs[imgs > 255] = 255.0
		imgs[imgs < 0] = 0.
		imgs = imgs / 127.5 - 1.
		sample['images'] = imgs
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)

		return sample

	def make_img_gt_pair(self, idx):
		"""
		Make the image-ground-truth pair
		"""

		img = imageio.imread(self.img_list[idx])
		imgs = [img]
		for i in range(self.frame_nums-1):
			imgs.append(img)

		gt = cv2.imread(self.labels[idx], 0)
		frame = img
		imgs = np.concatenate(imgs,axis=2)

		if self.inputRes is not None:

			imgs = cv2.resize(imgs, (self.inputRes[1],self.inputRes[0]))
			gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]),interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1],self.inputRes[0]))

		imgs = np.array(imgs, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)
		# normalize

		# pred_gt = frame / 127.5 - 1.
		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		return imgs, frame, gt, pred_gt


class DAVIS_Online_Dataset(Dataset):
	def __init__(self, train=True,
				 inputRes=None,
				 db_root_dir='/home/xk/Dataset/DAVIS/',
				 transform=None,
				 seq_name=None,
				 frame_nums=4):
		"""Loads image to label pairs for tool pose estimation
		db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
		"""
		self.train = train
		self.inputRes = inputRes
		self.db_root_dir = db_root_dir
		self.transform = transform
		self.seq_name = seq_name
		self.toTensor = tr.ToTensor()
		self.frame_nums = frame_nums
		# Initialize the per sequence images for online training
		names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
		img_list = list(map(lambda x: os.path.join(db_root_dir,'JPEGImages/480p/', str(seq_name), x), names_img))
		name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
		labels = list(map(lambda x: os.path.join(db_root_dir,'Annotations/480p/', str(seq_name), x), name_label))

		if self.train:
			img_list = [img_list[0]]
			labels = [labels[0]]

		assert (len(labels) == len(img_list))

		self.img_list = img_list
		self.labels = labels

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
		imgs,frame, gt, pred_gt = self.make_img_gt_pair(idx)

		sample = {'images': imgs, 'frame': frame, 'seg_gt': gt, 'pred_gt': pred_gt}

		if self.seq_name is not None:
			fname = os.path.join(self.seq_name, "%05d" % idx)
			sample['fname'] = fname

		if self.transform is not None:
			sample = self.transform(sample)

		imgs = sample['images']
		imgs[np.isnan(imgs)] = 0.
		imgs[imgs > 255] = 255.0
		imgs[imgs < 0] = 0.
		imgs = imgs / 127.5 - 1.
		sample['images'] = imgs
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)

		return sample

	def make_img_gt_pair(self, idx):
		"""
		Make the image-ground-truth pair
		"""
		assert self.frame_nums in [3,4,5,6], 'the frame nums is not support'
		if self.frame_nums ==3:
			if idx == 0:
				seq_path = [self.img_list[idx],self.img_list[idx],self.img_list[idx]]
			elif idx == 1:
				seq_path = [self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 2:
				seq_path = [self.img_list[idx-2], self.img_list[idx-1], self.img_list[idx]]
			else:
				seq_path = [self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx-1]]

		elif self.frame_nums ==4:
			if idx == 0:
				seq_path = [self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx]]
			elif idx == 1:
				seq_path = [self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 2:
				seq_path = [self.img_list[idx-2], self.img_list[idx-2], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 3:
				seq_path = [self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]
			else:
				seq_path = [self.img_list[idx - 4], self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx-1]]

		elif self.frame_nums == 5:
			if idx == 0:
				seq_path = [self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx]]
			elif idx == 1:
				seq_path = [self.img_list[idx-1],self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 2:
				seq_path = [self.img_list[idx-2],self.img_list[idx-2], self.img_list[idx-2], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 3:
				seq_path = [self.img_list[idx - 3],self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]
			elif idx == 4:
				seq_path = [self.img_list[idx - 4],self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]

			else:
				seq_path = [self.img_list[idx - 5],self.img_list[idx - 4], self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx-1]]

		elif self.frame_nums == 6:
			if idx == 0:
				seq_path = [self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx],self.img_list[idx]]
			elif idx == 1:
				seq_path = [self.img_list[idx-1],self.img_list[idx-1],self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 2:
				seq_path = [self.img_list[idx-2],self.img_list[idx-2],self.img_list[idx-2], self.img_list[idx-2], self.img_list[idx-1], self.img_list[idx]]
			elif idx == 3:
				seq_path = [self.img_list[idx - 3],self.img_list[idx - 3],self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]
			elif idx == 4:
				seq_path = [self.img_list[idx - 4],self.img_list[idx - 4],self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]
			elif idx == 5:
				seq_path = [self.img_list[idx - 5],self.img_list[idx - 4], self.img_list[idx - 3],
							self.img_list[idx - 2], self.img_list[idx - 1], self.img_list[idx]]
			else:
				seq_path = [self.img_list[idx - 6],self.img_list[idx - 5],self.img_list[idx - 4], self.img_list[idx - 3], self.img_list[idx - 2], self.img_list[idx-1]]


		imgs = []
		for i in range(self.frame_nums):
			img = imageio.imread(seq_path[i])
			imgs.append(img)

		gt = cv2.imread(self.labels[idx], 0)
		frame = imageio.imread(self.img_list[idx])
		imgs = np.concatenate(imgs,axis=2)

		if self.inputRes is not None:

			imgs = cv2.resize(imgs, (self.inputRes[1],self.inputRes[0]))
			gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]),interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1],self.inputRes[0]))


		imgs = np.array(imgs, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)
		# normalize

		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		return imgs, frame, gt, pred_gt

