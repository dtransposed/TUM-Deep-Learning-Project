#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN VGGNET11 FILE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
#import cv2
from vgg import VGG
from solver import Solver
from data import DataImport as DI
import os
#import matplotlib.pyplot as plt
#import time

#DATA IMPORT
imageDir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/all data' #directory to one folder where all training images are
imageDir1 = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/all data' #directory to one folder where all validation images are
data_dir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset'#this is actual directory of our dataset
train_dir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/train'
val_dir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/val'

train_mean=DI.get_mean(imageDir)
val_mean=DI.get_mean(imageDir1)
train_std=DI.get_std(imageDir)
val_std=DI.get_std(imageDir1)

#PREPROCESSING
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(224), #applies crop augumentation, arg is the size of the output
        transforms.RandomVerticalFlip(), #applies flip augumentation
        transforms.ToTensor(),#converts image to torch.FloatTensor
        #normalize according to RGB mean and standard deviation. args are list of 3 arguments, 
        #one for every colour
        transforms.Normalize(train_mean,train_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(val_mean,val_std)
    ]),
}

#function ImageFolder is pytorch's data reader
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
# =============================================================================
# train_data = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train']}
# val_data = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['val']}
# =============================================================================
#function DataLoader is pytorch's data loader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                            shuffle=True)
              for x in ['train', 'val']}
# =============================================================================
# dataloader_train = {x: torch.utils.data.DataLoader(train_data[x],
#                                             shuffle=True)
#               for x in ['train']}
# dataloader_val = {x: torch.utils.data.DataLoader(val_data[x],
#                                             shuffle=True)
#               for x in ['val']}
# =============================================================================

#class_names = dataloader_train.classes

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
#DI.imshow(out, train_mean, train_std)#, title=[class_names[x] for x in classes])

#num_train = 100
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False, num_workers=4,
#                                           sampler=OverfitSampler(num_train))
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, num_workers=4)

model = VGG()
#solver = Solver(optim_args={"lr": 1e-2})
#solver.train(model, dataloaders['train'], dataloaders['val'], log_nth=1, num_epochs=5)
if model.is_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
Solver.train_model(dataloaders, dataset_sizes, model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


