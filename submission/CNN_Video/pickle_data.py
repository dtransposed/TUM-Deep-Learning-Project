#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Precomputing output of CNN
'''

import torch
import numpy as np
import VideoDataLoader as VDL
import PretrainedCNN
import PredictionHead as PH
import VideoSolver as VS
import VideoClassifier as VC
import glob
import VideoPooling as VP
from torch.autograd import Variable
from matplotlib import pyplot as plt
import pickle

#path = '/home/peternagy96/Downloads/Frames_for _Damian' 
path = '/home/peternagy96/Project/big_dataset'

# dataloader
data_loader = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=1000, load_opt_flow=True)


for i, data in enumerate(data_loader):
    
    #pickle.dump(data,open('data_1500.p','wb'))
    N_train = int(data['targets'].shape[0]*0.8)
    N_val = int(data['targets'].shape[0]*0.2)
    data_train, data_val = VDL.split_dataset(data, size=0.2)
    
    pickle.dump(data_train['targets'],open('targets_5000_densenet121_train_class4.p','wb'))
    pickle.dump(data_val['targets'],open('targets_5000_densenet121_val_class4.p','wb'))
    
    train_loader = VDL.iterate_videos(data_train)
    val_loader = VDL.iterate_videos(data_val)
    
    #Get fully convolutional network for feature extraction on frame level
    pretrained_model = PretrainedCNN.Fully_Conv_Block('densenet121')
    
    if torch.cuda.is_available():
        pretrained_model.cuda()

    conv_out = torch.zeros(N_train,50176)
    for j, (inputs, targets) in enumerate(train_loader.__next__()):
        print('Training sample ' + str(j))
        #cast to variable
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        #Check whether cuda is available and utilize if possible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        out = pretrained_model(inputs)
        out = VP.average_pooling(out)
        print(out.data.shape)
        conv_out[j] = out.data
    
    pickle.dump(conv_out,open('conv_out_5000_densenet121_train_class4.p','wb'))
    
    conv_out = torch.zeros(N_val,50176)
    for j, (inputs, targets) in enumerate(val_loader.__next__()):
        print('Validation sample ' + str(j))
        #cast to variable
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        #Check whether cuda is available and utilize if possible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        out = pretrained_model(inputs)
        out = VP.average_pooling(out)
        
        conv_out[j] = out.data
    
    pickle.dump(conv_out,open('conv_out_5000_densenet121_val_class4.p','wb')) 

                    
                    
                    
                    
                    
                    
                    
                    
