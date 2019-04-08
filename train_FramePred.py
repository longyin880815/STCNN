from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
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
import torch.nn.functional as F
import math
from network.googlenet import Inception3
from network.framepred import FramePredNet
from dataloaders import custom_transforms as tr
from dataloaders import VID_dataloader as db
from mypath import Path

def main(args):
    # # Select which GPU, -1 if CPU
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    
    # # Setting other parameters
    resume_epoch = 0  # Default is 0, change if want to resume
    nEpochs = 100  # Number of epochs for training (500.000/2079)
    batch_size = 3
    snapshot = 10  # Store a model every snapshot epochs
    beta = 0.001
    margin = 0.3
    updateD = True
    updateG = True
    lr_G = 1e-7
    lr_D = 1e-4
    wd = 0.0002
    frame_nums = args.frame_nums

    save_root_dir = Path.save_root_dir()
    save_dir = os.path.join(save_root_dir,'FramePredModels','frame_nums_'+str(frame_nums))
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    # Network definition
    modelGName = 'NetG'
    modelDName = 'NetD'
    if resume_epoch == 0:
        netD = Inception3(num_classes=1, aux_logits=False, transform_input=True)
        initialize_netD(netD)
        netG = FramePredNet(input_frame_nums=frame_nums)
        initialize_netG(netG, input_frame_nums=frame_nums)
    else:
        netD = Inception3(num_classes=1, aux_logits=False, transform_input=True)

        netG = FramePredNet(input_frame_nums=frame_nums)

        print("Updating weights from: {}".format(
            os.path.join(save_dir, modelGName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
        netG.load_state_dict(
            torch.load(os.path.join(save_dir, modelGName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage))
        netD.load_state_dict(
            torch.load(os.path.join(save_dir, modelDName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage))


    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-parent')

    netD.to(device) # PyTorch 0.4.0 style
    netG.to(device)

    # Use the following optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D,weight_decay=wd)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G,weight_decay=wd)

    # Preparation of the data loaders
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.Normalize(),
                                              tr.ToTensor()])

    # Training dataset and its iterator
    db_train = db.VIDDataset(inputRes=(480,854), seqs_list_file=Path.VID_list_file(), transform=composed_transforms,
                             random_rev_thred=0.4, frame_len=frame_nums)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)

    num_img_tr = len(trainloader)

    criterion = nn.BCELoss().to(device)
    lp_function = nn.MSELoss().to(device)

    print("Training Network")
    real_label = torch.ones(batch_size).float().to(device)
    fake_label = torch.zeros(batch_size).float().to(device)
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['images'], sample_batched['gt']
            gts = F.upsample(gts, size=(120, 214), mode='bilinear', align_corners=True)
            if gts.size()[0] != batch_size:
                continue
            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)
            pred = netG.forward(inputs)

            D_real = netD(gts)
            errD_real = criterion(D_real, real_label)
            D_fake = netD(pred.detach())
            errD_fake = criterion(D_fake, fake_label)

            if updateD:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                # train with fake
                d_loss = errD_fake + errD_real
                d_loss.backward()
                optimizerD.step()

            if updateG:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                D_fake = netD(pred)
                errG = criterion(D_fake, real_label)
                lp_loss = lp_function(pred, gts)
                total_loss = lp_loss + beta * errG
                total_loss.backward()
                optimizerG.step()

            if (errD_fake.data < margin).all() or (errD_real.data < margin).all():
                updateD = False
            if (errD_fake.data > (1. - margin)).all() or (errD_real.data > (1. - margin)).all():
                updateG = False
            if not updateD and not updateG:
                updateD = True
                updateG = True

            if (ii + num_img_tr * epoch) % 5 == 0:
                print(
                    "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, im_loss: %.8f"
                    % (ii + num_img_tr * epoch, timeit.default_timer() - start_time, d_loss.item(), errG.item(), lp_loss.item())
                )
                print('updateD:', updateD, 'updateG:', updateG)
            if (ii + num_img_tr * epoch) % 10 == 0:
                writer.add_scalar('data/lp_loss_iter', lp_loss.item(), ii + num_img_tr * epoch)
                writer.add_scalar('data/adv_loss_iter', errG.item(), ii + num_img_tr * epoch)

            if (ii + num_img_tr * epoch) % 100 == 0:
                samples = pred[0, :, :, :].data.cpu().numpy()
                gt = gts[0, :, :, :].data.cpu().numpy()
                samples = samples.transpose([1, 2, 0])
                gt = gt.transpose([1, 2, 0])
                samples = np.concatenate((samples, gt), axis=0)
                print("Saving sample ...")
                samples = inverse_transform(samples)*255
                running_res_dir = os.path.join(save_dir, 'results')
                if not os.path.exists(running_res_dir):
                    os.makedirs(running_res_dir)
                imageio.imwrite(os.path.join(running_res_dir, "train_%s.png" % (ii + num_img_tr * epoch)), samples)

        # Print stuff
        print('[Epoch: %d, numImages: %5d]' % (epoch, (ii + 1)*batch_size))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time))
        # Save the model
        if (epoch % snapshot) == snapshot-1:
            print("save models")
            torch.save(netG.state_dict(), os.path.join(save_dir, modelGName + '_epoch-' + str(epoch) + '.pth'))
            torch.save(netD.state_dict(), os.path.join(save_dir, modelDName + '_epoch-' + str(epoch) + '.pth'))
    writer.close()

def inverse_transform(images):
    return (images+1.)/2.

def initialize_netG(net,input_frame_nums=4):
    print("Loading weights from PyTorch ResNet101")
    pretrained_dict = torch.load(os.path.join('./models', 'resnet101_pytorch.pth'))
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    weight = pretrained_dict['conv1.weight']
    pretrained_dict['conv1.weight'] = torch.cat([weight]*input_frame_nums,1)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)


def initialize_netD(net):
    print("Loading weights from PyTorch GoogleNet")
    pretrained_dict = torch.load(os.path.join('./models', 'inception_v3_google_pytorch.pth'))
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('fc' not in k) and ('AuxLogits' not in k)}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)



if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser for train frame predict")

    main_arg_parser.add_argument("--frame_nums", type=int, default=4,
                                 help="input frame nums")

    args = main_arg_parser.parse_args()
    main(args)