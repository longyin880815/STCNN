from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import socket
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio

from network.joint_pred_seg import SegBranch, SegDecoder,SegEncoder
from dataloaders import joint_transforms

from dataloaders import voc_msra_dataloader as db
from mypath import Path

# # Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	print('Using GPU: {} '.format(gpu_id))

# # Setting other parameters
last_iter = 0  # Default is 0, change if want to resume
iter_num = 12000
batch_size = 8
snapshot = 1000  # Store a model every snapshot epochs
lr = 1e-3
wd = 5e-4
lr_decay = 0.9
sidWeight = 0.5
modelName = 'Seg_Branch'

save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
	os.makedirs(os.path.join(save_dir))

save_model_dir = os.path.join(save_dir,modelName)
if not os.path.exists(save_model_dir):
	os.makedirs(os.path.join(save_model_dir))
log_dir = os.path.join(save_dir, 'SegBranch_runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir, comment='-parent')


def main():
	joint_transform = joint_transforms.Compose([
		joint_transforms.RandomCrop(300),
		joint_transforms.RandomHorizontallyFlip(),
		joint_transforms.RandomRotate(10)
	])
	img_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	target_transform = transforms.ToTensor()
	train_set = db.voc_msra_dataloadr(Path.MSRAdataset_dir(),Path.VOC_dir(), joint_transform, img_transform, target_transform)
	train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
	criterion = nn.BCEWithLogitsLoss().to(device)


	encoder = SegEncoder()
	initialize_SegEncoder(encoder)

	decoder = SegDecoder()
	net = SegBranch(net_enc=encoder,net_dec=decoder)
	net.to(device)
	optimizer = optim.SGD([
		{'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],'lr': 2 * lr},
		{'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],'lr': lr, 'weight_decay': wd}
		], momentum=0.9)

	if last_iter > 0:
		print ('training resumes from ' + str(last_iter))
		net.load_state_dict(torch.load(os.path.join(save_model_dir, modelName + '_epoch-' + str(last_iter-1) + '.pth')))

	curr_iter = last_iter

	while True:
		start_time = timeit.default_timer()
		for ii, sample_batched in enumerate(train_loader):
			optimizer.param_groups[0]['lr'] = 2 * lr*(1 - float(curr_iter) / iter_num) ** lr_decay
			optimizer.param_groups[1]['lr'] = lr * (1 - float(curr_iter) / iter_num) ** lr_decay


			inputs, gts = sample_batched['images'], sample_batched['gts']
			inputs.requires_grad_()

			inputs, gts = inputs.to(device), gts.to(device)
			pred = net.forward(inputs)
			optimizer.zero_grad()
			loss = criterion(pred[-1], gts)
			for i in reversed(range(len(pred) - 1)):
				loss = loss + 1 * criterion(pred[i], gts)
			# loss = criterion(pred, gts)
			loss.backward()
			optimizer.step()
			curr_iter += 1

			if curr_iter % 5 == 0:
				print(
					"Iters: [%2d] time: %4.4f, loss: %.8f"
					% (curr_iter, timeit.default_timer() - start_time, loss.item())
				)

			if curr_iter % 10 == 0:
				writer.add_scalar('data/loss_iter', loss.item(), curr_iter)

			if curr_iter % 100 == 1:

				inputs = inputs[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
				inputs = (inputs - inputs.min()) / max((inputs.max() - inputs.min()), 1e-8) * 255

				gt = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255
				gt = np.concatenate([gt, gt, gt], axis=2)

				samples = pred[-1][0, :, :, :].data.cpu().numpy()
				samples = 1 / (1 + np.exp(-samples))
				samples = samples.transpose([1, 2, 0]) * 255
				samples = np.concatenate([samples, samples, samples], axis=2)

				samples = np.concatenate((samples, gt, inputs), axis=0)

				print("Saving sample ...")
				# samples = inverse_transform(samples)*255
				running_res_dir = os.path.join(save_dir, modelName+'_results')
				if not os.path.exists(running_res_dir):
					os.makedirs(running_res_dir)
				imageio.imwrite(os.path.join(running_res_dir, "train_%s.png" % (curr_iter)), samples)

			# Save the model
			if (curr_iter % snapshot) == snapshot - 1:
				torch.save(net.state_dict(), os.path.join(save_model_dir, modelName + '_epoch-' + str(curr_iter) + '.pth'))
			if curr_iter == iter_num:
				return

def initialize_SegEncoder(net):
	print("Loading weights from PyTorch ResNet101")
	pretrained_dict = torch.load(os.path.join('./models', 'resnet101_pytorch.pth'))
	model_dict = net.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	net.load_state_dict(model_dict)

if __name__ == "__main__":
	main()