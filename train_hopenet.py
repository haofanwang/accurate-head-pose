import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet
import torch.utils.model_zoo as model_zoo

CUDA_LAUNCH_BLOCKING=1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=3, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.000001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW_multi', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='/mnt/hdfs-data-1/adas/haofan.wang/DMS-T-HeadPose/AFLW/', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='/mnt/hdfs-data-1/adas/haofan.wang/HeadPose_exp/deep-head-pose/code/tools/AFLW_train.txt', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=2, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll,
         model.fc_yaw_1, model.fc_pitch_1, model.fc_roll_1,
         model.fc_yaw_2, model.fc_pitch_2, model.fc_roll_2,
         model.fc_yaw_3, model.fc_pitch_3, model.fc_roll_3]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet.Multinet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 198)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Resize(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_multi':
        pose_dataset = datasets.Pose_300W_LP_multi(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI_multi':
        pose_dataset = datasets.BIWI_multi(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_multi':
        pose_dataset = datasets.AFLW_multi(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print 'Error: not a valid dataset name'
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in xrange(198)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                   lr = args.lr)

    print 'Ready to train network.'
    for epoch in range(num_epochs):
        for i, (images, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)
            
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)
            
            label_yaw_1 = Variable(labels_0[:,0]).cuda(gpu)
            label_pitch_1 = Variable(labels_0[:,1]).cuda(gpu)
            label_roll_1 = Variable(labels_0[:,2]).cuda(gpu)
            
            label_yaw_2 = Variable(labels_1[:,0]).cuda(gpu)
            label_pitch_2 = Variable(labels_1[:,1]).cuda(gpu)
            label_roll_2 = Variable(labels_1[:,2]).cuda(gpu)
            
            label_yaw_3 = Variable(labels_2[:,0]).cuda(gpu)
            label_pitch_3 = Variable(labels_2[:,1]).cuda(gpu)
            label_roll_3 = Variable(labels_2[:,2]).cuda(gpu)
            
            label_yaw_4 = Variable(labels_3[:,0]).cuda(gpu)
            label_pitch_4 = Variable(labels_3[:,1]).cuda(gpu)
            label_roll_4 = Variable(labels_3[:,2]).cuda(gpu)
                        
            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw,yaw_1,yaw_2,yaw_3,yaw_4, pitch,pitch_1,pitch_2,pitch_3,pitch_4, roll,roll_1,roll_2,roll_3,roll_4 = model(images)

            # Cross entropy loss
            loss_yaw,loss_yaw_1,loss_yaw_2,loss_yaw_3,loss_yaw_4 = criterion(yaw, label_yaw),criterion(yaw_1, label_yaw_1),criterion(yaw_2, label_yaw_2),criterion(yaw_3, label_yaw_3),criterion(yaw_4, label_yaw_4)
            loss_pitch,loss_pitch_1,loss_pitch_2,loss_pitch_3,loss_pitch_4 = criterion(pitch, label_pitch),criterion(pitch_1, label_pitch_1),criterion(pitch_2, label_pitch_2),criterion(pitch_3, label_pitch_3),criterion(pitch_4, label_pitch_4)
            loss_roll,loss_roll_1,loss_roll_2,loss_roll_3,loss_roll_4 = criterion(roll, label_roll),criterion(roll_1, label_roll_1),criterion(roll_2, label_roll_2),criterion(roll_3, label_roll_3),criterion(roll_4, label_roll_4)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) - 99
                        
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            total_loss_yaw = alpha * loss_reg_yaw + 7*loss_yaw + 5*loss_yaw_1 + 3*loss_yaw_2 + 1*loss_yaw_3 + 1*loss_yaw_4
            total_loss_pitch = alpha * loss_reg_pitch + 7*loss_pitch + 5*loss_pitch_1 + 3*loss_pitch_2 + 1*loss_pitch_3 + 1*loss_pitch_4
            total_loss_roll = alpha * loss_reg_roll + 7*loss_roll + 5*loss_roll_1 + 3*loss_roll_2 + 1*loss_roll_3 + 1*loss_pitch_4
            
            loss_seq = [total_loss_yaw, total_loss_pitch, total_loss_roll]
            grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, total_loss_yaw.item(), total_loss_pitch.item(), total_loss_roll.item()))
        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshots/setting28/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')