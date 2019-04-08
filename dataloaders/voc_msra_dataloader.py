import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import random
import cv2
import skimage.morphology as sm
from dataloaders import custom_transforms as tr
from torchvision import transforms

num_classes = 21
ignore_label = 255
root = '/home/xk/Dataset/VOC/'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
		   128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
		   64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
	palette.append(0)



def colorize_mask(mask):
	# mask: numpy array of the mask
	new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
	new_mask.putpalette(palette)

	return new_mask

def make_msra_dataset(root):
	img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
	return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png')) for img_name in img_list]


def make_voc_dataset(mode,root):
	assert mode in ['train', 'val', 'test']
	items = []
	if mode == 'train':
		img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
		mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'inst')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
		data_list.extend([l.strip('\n') for l in open(os.path.join(
			root, 'benchmark_RELEASE', 'dataset', 'val.txt')).readlines()])
		for it in data_list:
			item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
			items.append(item)
	elif mode == 'val':
		img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
		mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
		for it in data_list:
			item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
			items.append(item)
	else:
		img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
		for it in data_list:
			items.append((img_path, it))
	return items


class VOC(data.Dataset):
	def __init__(self, mode, inputRes,  transform=None):
		self.imgs = make_voc_dataset(mode)
		if len(self.imgs) == 0:
			raise RuntimeError('Found 0 images, please check the data set')
		self.mode = mode
		self.inputRes = inputRes
		self.transform = transform


	def __getitem__(self, index):

		img_path, mask_path = self.imgs[index]
		img = Image.open(img_path).convert('RGB')
		img = np.asarray(img,dtype=np.float32)
		img = img / 255
		img = np.subtract(img, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		img = np.true_divide(img, np.array([0.229, 0.224, 0.225], dtype=np.float32))

		gt = sio.loadmat(mask_path)['GTinst']['Segmentation'][0][0]
		gt = np.asarray(gt, dtype=np.float32)
		n = np.max(gt)
		a = random.randint(1, n)
		gt[gt != a] = 0.0
		gt[gt == a] = 1.0
		# gt[gt > 0] = 1.0

		if self.inputRes is not None:
			img = cv2.resize(img, (self.inputRes[1], self.inputRes[0]))
			gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]), interpolation=cv2.INTER_NEAREST)


		sample = {'images': img, 'gts': gt}
		if self.transform is not None:
			sample = self.transform(sample)

		return sample

	def __len__(self):
		return len(self.imgs)


class voc_msra_dataloadr(data.Dataset):
	# image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
	def __init__(self, msra_root,voc_root, joint_transform=None, transform=None, target_transform=None):

		self.imgs_msra = make_msra_dataset(msra_root)
		self.imgs_voc = make_voc_dataset('train',voc_root)
		self.imgs = []
		self.imgs.extend(self.imgs_msra)
		self.imgs.extend(self.imgs_voc)
		self.joint_transform = joint_transform
		self.transform = transform
		self.target_transform = target_transform
		self.msra_num = len(self.imgs_msra)

	def __getitem__(self, index):
		if index < self.msra_num:
			img_path, gt_path = self.imgs[index]
			img = Image.open(img_path).convert('RGB')
			target = Image.open(gt_path).convert('L')
		else:
			img_path, mask_path = self.imgs[index]
			img = Image.open(img_path).convert('RGB')
			target = sio.loadmat(mask_path)['GTinst']['Segmentation'][0][0]
			target = np.asarray(target, dtype=np.float32)

			target[target > 0] = 255.0
			target = Image.fromarray(target).convert('L')

		if self.joint_transform is not None:
			img, target = self.joint_transform(img, target)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		target = target.numpy()
		sample = {'images':img, 'gts': target}
		return sample

	def __len__(self):
		return len(self.imgs)




if __name__ == '__main__':
	composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
											  tr.ScaleNRotate(rots=(-10, 10), scales=(.75, 1.25)),
											  tr.ToTensor()])
	train_set = VOC('train',inputRes=(512,512),transform=composed_transforms)
	train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True)
	for ii, sample_batched in enumerate(train_loader):
		img,mask = sample_batched
		break
