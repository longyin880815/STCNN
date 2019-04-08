from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.layers import interp_surgery



def crop_like(x, target):
	if x.size()[2:] == target.size()[2:]:
		return x
	else:
		height = target.size()[2]
		width = target.size()[3]
		crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
		crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
	# fixed indexing for PyTorch 0.4
	return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


def ConvBatchNormReLU(in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1,relu=True,bias = False):
	if relu == True:
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
				stride=stride,padding=padding,dilation=dilation,bias=bias),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True))
	else:
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
					  stride=stride, padding=padding, dilation=dilation, bias=bias),
			nn.BatchNorm2d(out_channels),
			)

class RegionBottleneck(nn.Module):
	def __init__(self, planes):
		super(RegionBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu1 = nn.ReLU(inplace=True)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

	def forward(self, input):
		x = input[0]
		mask = input[1]
		residual = x
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.conv1(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.conv2(x)
		x = x*mask
		out = residual + x
		return out

class Refine(nn.Module):
	def __init__(self, planes, step=1):
		super(Refine, self).__init__()
		self.rgConv_list = nn.ModuleList()
		for i in range(step):
			self.rgConv_list.append(RegionBottleneck(planes))

	def forward(self, input):
		x = input[0]
		mask = input[1]
		for i in range(len(self.rgConv_list)):
			x = self.rgConv_list[i]([x,mask])
		return x

class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
		self.bn1 = nn.BatchNorm2d(planes)
		padding = 1
		if dilation == 2:
			padding = 2
		elif dilation == 4:
			padding = 4
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
								padding=padding, bias=False, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out




class SegEncoder(nn.Module):
	def __init__(self, multi_grid=[1, 2, 1],freez_bn=True):
		self.inplanes = 64
		layers = [3, 4, 23, 3]
		block = Bottleneck
		super(SegEncoder, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1,dilation=2)
		# self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation__=4)
		self.layer4 = self._make_layer_mg(block, 512, layers[3], stride=1, dilation=2, mg=multi_grid)


		print("Initializing weights..")
		self._initialize_weights()
		if freez_bn == True:
			self.freeze_bn()

	def forward(self, x, return_feature_maps=False):

		conv_out = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x); conv_out.append(x);
		x = self.layer2(x); conv_out.append(x);
		x = self.layer3(x); conv_out.append(x);
		x = self.layer4(x); conv_out.append(x);

		if return_feature_maps:
			return conv_out
		return [x]

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# m.weight.data.normal_(0, math.sqrt(2. / n))
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			# elif isinstance(m, nn.ConvTranspose2d):
			#     m.weight.data.zero_()
			#     m.weight.data = interp_surgery(m)

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
			nn.Conv2d(self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion)
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation))

		return nn.Sequential(*layers)

	def _make_layer_mg(self, block, planes, blocks=3, stride=1, dilation=2, mg=[1, 2, 1]):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion)
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=dilation*mg[0], downsample=downsample))
		self.inplanes = planes * block.expansion
		layers.append(block(self.inplanes, planes, dilation=dilation*mg[1]))
		layers.append(block(self.inplanes, planes, dilation=dilation*mg[2]))
		return nn.Sequential(*layers)

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()


class SegDecoder(nn.Module):
	def __init__(self, num_class=1, fc_dim=2048,
				 use_softmax=False, pool_scales=(1, 2, 3, 6),
				 fpn_inplanes=(256,512,1024,2048), fpn_dim=256,freez_bn=True):
		super(SegDecoder, self).__init__()
		self.use_softmax = use_softmax

		# PPM Module
		self.ppm_pooling = []
		self.ppm_conv = []

		for scale in pool_scales:
			self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
			self.ppm_conv.append(nn.Sequential(
				nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU(inplace=True)
			))
		self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
		self.ppm_conv = nn.ModuleList(self.ppm_conv)
		self.ppm_last_conv = ConvBatchNormReLU(fc_dim + len(pool_scales)*512, fpn_dim)

		# FPN Module
		self.fpn_in = []
		for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
			self.fpn_in.append(nn.Sequential(
				nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
				nn.BatchNorm2d(fpn_dim),
				nn.ReLU(inplace=True)
			))
		self.fpn_in = nn.ModuleList(self.fpn_in)

		self.fpn_out = []
		for i in range(len(fpn_inplanes) - 1): # skip the top layer
			self.fpn_out.append(nn.Sequential(
				ConvBatchNormReLU(fpn_dim, fpn_dim)
			))
		self.fpn_out = nn.ModuleList(self.fpn_out)

		self.score_out = []
		for i in range(len(fpn_inplanes)):  # skip the top layer
			self.score_out.append(nn.Sequential(
				ConvBatchNormReLU(fpn_dim, fpn_dim),
				nn.Conv2d(fpn_dim, num_class, 1),
			))
		self.score_out = nn.ModuleList(self.score_out)

		self.upscale = []
		for i in range(len(fpn_inplanes) - 1):
			self.upscale.append(nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, bias=False))
		self.upscale = nn.ModuleList(self.upscale)
		self.att_out = []

		self.att_out = []
		for i in range(len(fpn_inplanes)-1):  # skip the top layer
			self.att_out.append(Refine(fpn_dim, 1))
		self.att_out = nn.ModuleList(self.att_out)

		self.conv_last = nn.Sequential(
			ConvBatchNormReLU(len(fpn_inplanes) * fpn_dim, fpn_dim),
			nn.Conv2d(fpn_dim, num_class, kernel_size=1)
		)

		if freez_bn == True:
			self.freeze_bn()

	def forward(self, conv_out, segSize=None):
		results = []
		conv5 = conv_out[-1]
		input_size = conv5.size()
		ppm_out = [conv5]
		for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
			ppm_out.append(pool_conv(nn.functional.upsample(
				pool_scale(conv5),
				(input_size[2], input_size[3]),
				mode='bilinear', align_corners=False)))
		ppm_out = torch.cat(ppm_out, 1)
		f = self.ppm_last_conv(ppm_out)

		seg_res = self.score_out[-1](f)
		# seg_res_up = F.upsample(seg_res, size=conv_out[0].size()[2:], mode='bilinear', align_corners=False)
		results.append(seg_res)

		fpn_feature_list = [f]
		for i in reversed(range(len(conv_out) - 1)):
			conv_x = conv_out[i]
			conv_x = self.fpn_in[i](conv_x) # lateral branch

			# f = F.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
			f = crop_like(self.upscale[i](f), conv_x)
			f = conv_x + f
			f_1 = self.fpn_out[i](f)

			seg_res = F.upsample(seg_res, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
			seg_res = F.sigmoid(seg_res)
			f_1 = self.att_out[i]([f_1, seg_res])
			seg_res = self.score_out[i](f_1)
			# seg_res_up = F.upsample(seg_res, size=conv_out[0].size()[2:], mode='bilinear', align_corners=False)
			results.append(seg_res)

			fpn_feature_list.append(f_1)

		fpn_feature_list.reverse() # [P2 - P5]
		output_size = fpn_feature_list[0].size()[2:]
		fusion_list = [fpn_feature_list[0]]
		for i in range(1, len(fpn_feature_list)):
			fusion_list.append(nn.functional.upsample(
				fpn_feature_list[i],
				output_size,
				mode='bilinear', align_corners=False))
		fusion_out = torch.cat(fusion_list, 1)
		x = self.conv_last(fusion_out)
		results.append(x)

		return results

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()


class FramePredEncoder(nn.Module):
	def __init__(self,frame_nums=4):
		self.inplanes = 64
		layers = [3, 4, 23, 3]
		block = Bottleneck
		super(FramePredEncoder, self).__init__()
		self.conv1 = nn.Conv2d(frame_nums*3, 64, kernel_size=7, stride=2, padding=3,
							 bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		# maxpool different from pytorch-resnet, to match tf-faster-rcnn
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		print("Initializing weights..")
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				m.weight.data.zero_()
				m.weight.data = interp_surgery(m)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
			  nn.Conv2d(self.inplanes, planes * block.expansion,
						kernel_size=1, stride=stride, bias=False),
			  nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, return_feature_maps=False):
		conv_out = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x); conv_out.append(x);
		x = self.layer2(x); conv_out.append(x);
		x = self.layer3(x); conv_out.append(x);
		x = self.layer4(x); conv_out.append(x);
		if return_feature_maps:
			return conv_out
		return [x]


class FramePredDecoder(nn.Module):
	def __init__(self):
		super(FramePredDecoder, self).__init__()
		# Decoder
		self.convC_1 = nn.Conv2d(512 * 4, 512 * 2, kernel_size=1, stride=1)
		self.convC_2 = nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1)
		self.convC_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

		self.de_layer1 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, stride=1, padding=1),
									   nn.BatchNorm2d(512),
									   # nn.ReLU(inplace=True),
									   nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, bias=False))

		self.de_layer2 = nn.Sequential(nn.Conv2d(512 * 2, 256, kernel_size=3, stride=1, padding=1),
									   nn.BatchNorm2d(256),
									   # nn.ReLU(inplace=True),
									   nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False))

		self.de_layer3 = nn.Sequential(nn.Conv2d(256 * 2, 64, kernel_size=3, stride=1, padding=1),
									   nn.BatchNorm2d(64),
									   # nn.ReLU(inplace=True),
									   nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, bias=False),
									   nn.Conv2d(64, 3, kernel_size=1, stride=1)
									   )

		print("Initializing weights..")
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				m.weight.data.zero_()
				m.weight.data = interp_surgery(m)

	def forward(self, conv_feats, return_feature_maps=False):
		conv_out = []
		x4 = self.convC_1(conv_feats[-1])
		out1 = self.de_layer1(x4)
		out1 = crop_like(out1, conv_feats[-2]);conv_out.append(out1);
		x3 = self.convC_2(conv_feats[-2])
		out2 = self.de_layer2(torch.cat((out1, x3), 1))
		out2 = crop_like(out2, conv_feats[-3]);conv_out.append(out2);
		x2 = self.convC_3(conv_feats[-3])
		out3 = torch.cat((out2, x2), 1)
		modulelist = list(self.de_layer3.modules())
		for l in modulelist[1:-1]:
			out3 = l(out3)
		out3 = crop_like(out3, conv_feats[-4]);conv_out.append(out3);
		out4 = modulelist[-1](out3)
		# out3 = F.upsample(out3, size = img_size, mode='bilinear',align_corners=True)
		pred = F.tanh(out4)

		if return_feature_maps:
			return pred, conv_out
		return pred

class JointSegDecoder(nn.Module):
	def __init__(self, num_class=1, fc_dim=2048, pool_scales=(1, 2, 3, 6),
				 fpn_inplanes=(256,512,1024,2048), fpn_dim=256,freez_bn=True):
		super(JointSegDecoder, self).__init__()

		# PPM Module
		self.ppm_pooling = []
		self.ppm_conv = []

		for scale in pool_scales:
			self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
			self.ppm_conv.append(nn.Sequential(
				nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU(inplace=True)
			))
		self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
		self.ppm_conv = nn.ModuleList(self.ppm_conv)
		self.ppm_last_conv = ConvBatchNormReLU(fc_dim + len(pool_scales)*512, fpn_dim)

		# FPN Module
		self.fpn_in = []
		for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
			self.fpn_in.append(nn.Sequential(
				nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
				nn.BatchNorm2d(fpn_dim),
				nn.ReLU(inplace=True)
			))
		self.fpn_in = nn.ModuleList(self.fpn_in)

		fpn_out = []
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+64, fpn_dim)))
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+256, fpn_dim)))
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+512, fpn_dim)))
		self.joint_fpn_out = nn.ModuleList(fpn_out)

		self.score_out = []
		for i in range(len(fpn_inplanes)):  # skip the top layer
			self.score_out.append(nn.Sequential(
				ConvBatchNormReLU(fpn_dim, fpn_dim),
				nn.Conv2d(fpn_dim, num_class, 1),
			))
		self.score_out = nn.ModuleList(self.score_out)

		self.upscale = []
		for i in range(len(fpn_inplanes) - 1):
			self.upscale.append(nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, bias=False))
		self.upscale = nn.ModuleList(self.upscale)

		self.att_out = []
		for i in range(len(fpn_inplanes) - 1):  # skip the top layer
			self.att_out.append(Refine(fpn_dim, 1))
		self.att_out = nn.ModuleList(self.att_out)

		self.conv_last = nn.Sequential(
			ConvBatchNormReLU(len(fpn_inplanes) * fpn_dim, fpn_dim),
			nn.Conv2d(fpn_dim, num_class, kernel_size=1)
		)

		if freez_bn == True:
			self.freeze_bn()

	def forward(self, conv_out, pred_de_feats):
		results = []
		conv5 = conv_out[-1]
		input_size = conv5.size()
		ppm_out = [conv5]
		for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
			ppm_out.append(pool_conv(nn.functional.upsample(
				pool_scale(conv5),
				(input_size[2], input_size[3]),
				mode='bilinear', align_corners=False)))
		ppm_out = torch.cat(ppm_out, 1)
		f = self.ppm_last_conv(ppm_out)

		seg_res = self.score_out[-1](f)
		results.append(seg_res)

		fpn_feature_list = [f]
		###########
		pred_de_feats[0] = nn.functional.upsample(pred_de_feats[0], size=f.size()[2:], mode='bilinear', align_corners=False)
		###########

		for i in reversed(range(len(conv_out) - 1)):
			conv_x = conv_out[i]
			conv_x = self.fpn_in[i](conv_x) # lateral branch

			# f = F.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
			f = crop_like(self.upscale[i](f), conv_x)
			f = conv_x + f
			###########
			pred_de_feats[2 - i] = crop_like(pred_de_feats[2-i], f)
			joint_feature = torch.cat([f, pred_de_feats[2-i]], 1)
			joint_feature = self.joint_fpn_out[i](joint_feature)

			seg_res = F.upsample(seg_res, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
			seg_res = F.sigmoid(seg_res)
			joint_feature = self.att_out[i]([joint_feature, seg_res])
			seg_res = self.score_out[i](joint_feature)
			results.append(seg_res)

			fpn_feature_list.append(joint_feature)
			###########
		fpn_feature_list.reverse() # [P2 - P5]
		output_size = fpn_feature_list[0].size()[2:]
		fusion_list = [fpn_feature_list[0]]
		for i in range(1, len(fpn_feature_list)):
			fusion_list.append(nn.functional.upsample(
				fpn_feature_list[i],
				output_size,
				mode='bilinear', align_corners=False))
		fusion_out = torch.cat(fusion_list, 1)
		x = self.conv_last(fusion_out)
		results.append(x)
		return results


	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

class STCNN(nn.Module):
	def __init__(self, pred_enc, seg_enc, pred_dec, j_seg_dec):
		super(STCNN, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		self.seg_encoder = seg_enc
		self.seg_decoder = j_seg_dec

	def forward(self, seq, frame):
		pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
		pred, pred_de_feats = self.pred_decoder(pred_en_feats,return_feature_maps=True)
		pred_feats = pred_de_feats
		for i in range(len(pred_de_feats)):
			pred_feats[i] = (pred_feats[i].detach())
		seg_en_feats = self.seg_encoder(frame, return_feature_maps=True)

		seg_res = self.seg_decoder(seg_en_feats, pred_feats)

		if isinstance(seg_res,list):
			for i in range(len(seg_res)):
				seg_res[i] = F.upsample(seg_res[i], size=frame.size()[2:], mode='bilinear', align_corners=False)
		else:
			seg_res = F.upsample(seg_res, size=frame.size()[2:], mode='bilinear', align_corners=False)

		return seg_res,pred

class PredBranch(nn.Module):
	def __init__(self, pred_enc, pred_dec):
		super(PredBranch, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec

	def forward(self, seq):
		pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
		pred = self.pred_decoder(pred_en_feats, return_feature_maps=False)
		return pred


class SegBranch(nn.Module):
	def __init__(self, net_enc, net_dec):
		super(SegBranch, self).__init__()
		self.encoder = net_enc
		self.decoder = net_dec

	def forward(self, data):
		feats = self.encoder(data, return_feature_maps=True)

		pred = self.decoder(feats)
		if isinstance(pred,list):
			for i in range(len(pred)):
				pred[i] = F.upsample(pred[i], size=data.size()[2:], mode='bilinear', align_corners=False)
		else:
			pred = F.upsample(pred, size=data.size()[2:], mode='bilinear', align_corners=False)
		return pred