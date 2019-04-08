# Package Includes
from __future__ import division

import argparse
import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
# PyTorch includes
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Custom includes
from dataloaders import DAVIS_dataloader as db
from dataloaders.DAVIS_dataloader import im_normalize
from dataloaders import custom_transforms as tr

import cv2
import scipy.misc as sm
from network.joint_pred_seg import FramePredDecoder,FramePredEncoder,SegEncoder,JointSegDecoder,STCNN


from mypath import Path


main_arg_parser = argparse.ArgumentParser(description="parser for train frame predict")
main_arg_parser.add_argument("--frame_nums", type=int, default=4,
							 help="input frame nums")
args = main_arg_parser.parse_args()

db_root_dir = Path.db_root_dir()
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
	os.makedirs(os.path.join(save_dir))

parentModelName = 'STCNN_frame_'+str(args.frame_nums)
save_model_dir = os.path.join(save_dir, parentModelName)
save_online_model_dir = os.path.join(save_dir, parentModelName)
if not os.path.exists(save_online_model_dir):
	os.makedirs(os.path.join(save_online_model_dir))

vis_res = 1  # Visualize the results?
nEpochs = 6
snapshot = nEpochs  # Store a model every snapshot epochs
parentEpoch = 10
# Parameters in p are used for the name of the model
trainBatch = 1  # Number of Images in each mini-batch
seed = 0
seg_lr = 1e-4
wd = 5e-4
resume = 0

# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")


lp_function = nn.MSELoss().to(device)
criterion = nn.BCELoss().to(device)
seg_criterion = nn.BCEWithLogitsLoss().to(device)

# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
										  tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25))
										  ])

fname = 'val'
with open(os.path.join(db_root_dir, 'ImageSets/2016/', fname + '.txt')) as f:
	seqnames = f.readlines()


for i in range(len(seqnames)):
	seq_name = seqnames[i].strip()
	seg_enc = SegEncoder()
	pred_enc = FramePredEncoder(frame_nums=args.frame_nums)
	pred_dec = FramePredDecoder()
	j_seg_dec = JointSegDecoder()
	net = STCNN(pred_enc, seg_enc, pred_dec, j_seg_dec)
	if resume != 0:
		net.load_state_dict(
			torch.load(os.path.join(save_online_model_dir,  seq_name+ '_epoch-' + str(resume - 1) + '.pth'),
					   map_location=lambda storage, loc: storage))
	else:
		net.load_state_dict(torch.load(os.path.join(save_model_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
									   map_location=lambda storage, loc: storage))
	# Logging into Tensorboard
	log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+'-'+seq_name)
	writer = SummaryWriter(log_dir=log_dir)

	net.to(device)  # PyTorch 0.4.0 style


	# Use the following optimizer
	optimizer = optim.SGD([
		{'params': [param for name, param in net.seg_encoder.named_parameters()], 'lr': seg_lr},
		{'params': [param for name, param in net.seg_decoder.named_parameters()], 'lr': seg_lr},
	], weight_decay=wd, momentum=0.9)
	# fix the pred network
	for param in net.pred_encoder.parameters():
		param.requires_grad = False
	for param in net.pred_decoder.parameters():
		param.requires_grad = False


	# Training dataset and its iterator
	db_train = db.DAVIS_First_Frame_Dataset(train=True,inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms,
											seq_name=seq_name,frame_nums=args.frame_nums)
	trainloader = DataLoader(db_train, batch_size=trainBatch, shuffle=True, num_workers=4)

	# Testing dataset and its iterator
	db_test = db.DAVIS_Online_Dataset(train=False, db_root_dir=db_root_dir, transform=None, seq_name=seq_name,
										  frame_nums=args.frame_nums)
	testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)

	num_img_tr = len(trainloader)
	num_img_ts = len(testloader)

	aveGrad = 0

	print("Start of Online Training, sequence: " + seq_name)
	start_time = timeit.default_timer()
	# Main Training and Testing Loop
	for epoch in range(resume, nEpochs):
		# One training epoch
		running_loss_tr = 0

		for ii, sample_batched in enumerate(trainloader):

			seqs, frames, gts, pred_gts = sample_batched['images'], sample_batched['frame'], sample_batched['seg_gt'], \
										  sample_batched['pred_gt']

			# Forward-Backward of the mini-batch

			frames.requires_grad_()

			seqs, frames, pred_gts, gts = seqs.to(device), frames.to(device), pred_gts.to(device)\
											  , gts.to(device)
			pred_gts = F.upsample(pred_gts, size=(100, 178), mode='bilinear', align_corners=True)
			pred_gts = pred_gts.detach()
			seg_res, pred = net.forward(seqs, frames)
			optimizer.zero_grad()

			seg_loss = seg_criterion(seg_res[-1], gts)
			seg_loss.backward()
			optimizer.step()

		stop_time = timeit.default_timer()

		# Print stuff
		print('[Epoch: %d]' % (epoch+1))
		print('seg_Loss: %f' % (seg_loss))
		writer.add_scalar('data/total_loss_epoch', seg_loss.item(), epoch)

		if (epoch % snapshot) == snapshot - 1 and epoch != 0:
			torch.save(net.state_dict(), os.path.join(save_online_model_dir, seq_name + '_epoch-'+str(epoch) + '.pth'))


	print('Online training time: ' + str(stop_time - start_time))


	# Testing Phase
	if vis_res:
		import matplotlib.pyplot as plt
		plt.close("all")
		plt.ion()
		f, ax_arr = plt.subplots(1, 4)

	save_dir_res = os.path.join(save_dir, parentModelName+'_Results', seq_name)
	if not os.path.exists(save_dir_res):
		os.makedirs(save_dir_res)


	print('Testing Network')
	with torch.no_grad():  # PyTorch 0.4.0 style
		# Main Testing Loop
		for ii, sample_batched in enumerate(testloader):

			seqs, frames, gts, pred_gts, fname = sample_batched['images'], sample_batched['frame'], sample_batched['seg_gt'], \
										  sample_batched['pred_gt'],sample_batched['fname']
			seqs, frames, gts, pred_gts = seqs.to(device), frames.to(device), gts.to(device), pred_gts.to(device)


			if ii == 0:
				mask_ = gts.cpu().numpy()[0, 0, :, :]
			else:
				pred[pred > 0.4] = 1
				pred[pred <= 0.4] = 0
				mask_ = pred

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
			mask__ = cv2.dilate(mask_, kernel)

			outputs, pred = net.forward(seqs, frames)

			for jj in range(int(seqs.size()[0])):
				pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
				pred = 1 / (1 + np.exp(-pred))
				pred = np.squeeze(pred)
				if np.sum(mask__ > 0) > 50:
					pred = pred * mask__
				# Save the result, attention to the index jj

				sm.imsave(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

				if vis_res:
					img_ = np.transpose(frames.cpu().numpy()[jj, :, :, :], (1, 2, 0))
					gt_ = np.transpose(gts.cpu().numpy()[jj, :, :, :], (1, 2, 0))
					gt_ = np.squeeze(gt_)
					# Plot the particular example
					ax_arr[0].cla()
					ax_arr[1].cla()
					ax_arr[2].cla()

					ax_arr[0].set_title('Input Image')
					ax_arr[1].set_title('Ground Truth')
					ax_arr[2].set_title('Detection')

					ax_arr[0].imshow(im_normalize(img_))
					ax_arr[1].imshow(gt_)
					ax_arr[2].imshow(im_normalize(pred))

					plt.pause(0.001)

	writer.close()
