#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import VideoDataLoader_fixed as VDL
import train_fixed
import CNN_fixed
import datetime
import pickle

# =============================================================================
# At this moment it is the same as the general main file
# =============================================================================

plt.ion()   # interactive mode

use_gpu = torch.cuda.is_available()

path = '/home/peternagy96/Project/big_dataset/'
data = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=400)

#data = pickle.load(open('datasets/hypertune1000_Jan_23_16:29.p','rb'))

print('orig size:')
print(data['data'].shape)
print('orig size:')
print(data['targets'].shape)

data_train, data_val = VDL.split_dataset(data, size=0.2)

N_train = data_train['targets'].shape[0]
N_val = data_val['targets'].shape[0]

data_train['data'] = np.swapaxes(data_train['data'],2,3)
data_train['data'] = np.swapaxes(data_train['data'],1,2)

data_val['data'] = np.swapaxes(data_val['data'],2,3)
data_val['data'] = np.swapaxes(data_val['data'],1,2)

data_train['data'] = torch.from_numpy(data_train['data']).type(torch.FloatTensor)
data_train['targets'] = torch.from_numpy(data_train['targets'])

data_val['data'] = torch.from_numpy(data_val['data']).type(torch.FloatTensor)
data_val['targets'] = torch.from_numpy(data_val['targets'])

train1 = TensorDataset(data_train['data'], data_train['targets'])
val1 = TensorDataset(data_val['data'], data_val['targets'])

train_loader = torch.utils.data.DataLoader(train1, batch_size=500,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val1, batch_size=500,shuffle=True, num_workers=0)


lr_values = np.array([0.00005])#, 0.0001, 0.0002]) #for the fc net

best_lr = []
best_acc = 0

k = 0

#TUNING

for lr in lr_values:
        k += 1
        print("##### ROUND "+str(k)+" #####")
        model_ft = models.densenet161(pretrained=True)
        for param in model_ft.features.parameters():
                    param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 5)
        
        #LOAD MODEL PARAMETERS IF NEEDED
        #model_ft.load_state_dict(torch.load("models/test.pth.tar"))
        
        if use_gpu:
            model_ft = model_ft.cuda()
            print("\nModel loaded in CUDA")
            
        print("\nlr: "+str(lr))
        
        
        optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        solver = train_fixed.Solver(optimizer=optimizer_ft)
        
        model_ft = solver.train_model(model_ft, train_loader, val_loader, N_train, N_val, num_epochs=50)
        
        for acc in solver.val_acc_history:
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_model = model_ft
                
        
        #VISUALIZE
        plt.subplot(2, 1, 1)
        plt.plot(solver.train_loss_history, 'o')
        plt.plot(solver.val_loss_history, 'o')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        
        
        plt.subplot(2, 1, 2)
        plt.plot(solver.train_acc_history, '-o')
        plt.plot(solver.val_acc_history, '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('figures/lrtestfinal_'+str(k)+'.png', bbox_inches='tight', dpi=200)
        plt.show()
        
print("BEST VAL ACCURACY: "+str(best_acc))
print("with lr: "+str(lr))

acc_history = {'train_loss': solver.train_loss_history,
               'train_acc': solver.train_acc_history,
               'val_loss': solver.val_loss_history,
               'val_acc': solver.val_acc_history}
pickle.dump(acc_history, open('figures/lr00002'+'.p','wb') )
#Save the model
torch.save(best_model.state_dict(), "models/lr00002.pth.tar")
print("\n==========\nModel successfully saved!\n==========")

