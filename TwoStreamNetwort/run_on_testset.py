#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:04:00 2018

@author: peternagy96
"""

'''
This script runs the final tests on the two testsets
'''


import VideoDataLoader1 as VDL
import pickle
import numpy as np
import torch
from torch.autograd import Variable

model_rgb = torch.load('model_rgb_final')
model_opt = torch.load('model_flow_final')

train_set, test_set = pickle.load(open('train_test_paths.p','rb'))

test_loader = VDL.load_videos_path(test_set,batchsize=1)

if torch.cuda.is_available():
    model_rgb = model_rgb.cuda()
    model_flow = model_flow.cuda()
    print("\nModel loaded in CUDA")

pred_scores = []

# compute prediction
for j, (inputs_rgb, inputs_opt, targets) in enumerate(test_loader.__next__()):
    
    print('getting batch ' + str(j))     
           
    targets = torch.from_numpy(targets)
    targets = targets = Variable(targets)
    if torch.cuda.is_available():
        targets = targets.cuda()
    i = 0
    for inp in inputs_rgb:
        inputs = torch.from_numpy(np.array(inp))
        inputs_opt = torch.from_numpy(np.array(inputs_opt[i]))
        
        inputs = Variable(inputs)
        inputs_opt = Variable(inputs_opt)
        
        #Check whether cuda is available and utilize if possible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        inputs_opt = inputs_opt.view(7,10,224,224).type(torch.cuda.FloatTensor)
        outputs_opt = model_opt.forward(inputs_opt)
        
        outputs = model_rgb.forward(inputs)
        
        if len(outputs.cpu().data.numpy().shape) == 1:
            outputs = outputs.view(1,outputs.shape[0])
        if len(outputs_opt.cpu().data.numpy().shape) == 1:
            outputs_opt = outputs_opt.view(1,outputs.shape[0])
            
        # max probability prediction
        if torch.max(outputs,1)[0].cpu().data.numpy() >= torch.max(outputs_opt,1)[0].cpu().data.numpy():
            _, preds = torch.max(outputs,1)
        else:
            _, preds = torch.max(outputs_opt,1)
        
        # zero for wrong and 1 for correct classification
        scores = (preds.cpu() == targets[i].cpu()).data.numpy()
        
        pred_scores.append(scores)
        
        i+=1

print(np.mean(pred_scores))


