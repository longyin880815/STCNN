from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from layers.layers import center_crop, interp_surgery
# from mypath import Path





class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
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


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    # maxpool different from pytorch-resnet, to match tf-faster-rcnn
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x





class FramePredNet(nn.Module):
    def __init__(self,input_frame_nums=4):
      self.inplanes = 64
      layers = [3, 4, 23, 3]
      block = Bottleneck
      super(FramePredNet, self).__init__()
      self.conv1 = nn.Conv2d(input_frame_nums*3, 64, kernel_size=7, stride=2, padding=3,
                             bias=False)
      self.bn1 = nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      # maxpool different from pytorch-resnet, to match tf-faster-rcnn
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      self.layer1 = self._make_layer(block, 64, layers[0])
      self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
      self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
      self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

      # Decoder
      self.convC_1 = nn.Conv2d(512*4, 512*2, kernel_size=1, stride=1)
      self.convC_2 = nn.Conv2d(512*2, 512, kernel_size=1, stride=1)
      self.convC_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

      self.de_layer1 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, bias=False))

      self.de_layer2 = nn.Sequential(nn.Conv2d(512*2, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False))

      self.de_layer3 = nn.Sequential(nn.Conv2d(256*2, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
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




    def _make_layer(self, block, planes, blocks, stride=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion),
        )

      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

      return nn.Sequential(*layers)



    def forward(self, x):

      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x1 = self.layer1(x)
      x2 = self.layer2(x1)
      x3 = self.layer3(x2)
      x4 = self.layer4(x3)

      x4 = self.convC_1(x4)
      out1 = self.de_layer1(x4)
      out1 = crop_like(out1,x3)
      x3 = self.convC_2(x3)
      out2 = self.de_layer2(torch.cat((out1,x3),1))
      out2 = crop_like(out2,x2)
      x2 = self.convC_3(x2)
      out3 = self.de_layer3(torch.cat((out2,x2),1))
      out3 = crop_like(out3, x1)
      out3 = F.tanh(out3)

      return out3


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


