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

plt.ion()   # interactive mode

# =============================================================================
# Used to run the model on the test dataset.
# =============================================================================

use_gpu = torch.cuda.is_available()

path = '/home/peternagy96/Project/small_testset/'
data = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=10)

print('orig size:')
print(data['data'].shape)
print('orig size:')
print(data['targets'].shape)

N_test = data['targets'].shape[0]

data['data'] = np.swapaxes(data['data'],2,3)
data['data'] = np.swapaxes(data['data'],1,2)

data['data'] = torch.from_numpy(data['data']).type(torch.FloatTensor)
data['targets'] = torch.from_numpy(data['targets'])

data1 = TensorDataset(data['data'], data['targets'])

test_loader = torch.utils.data.DataLoader(data1, batch_size=500,shuffle=True, num_workers=0)

best_lr = []
best_acc = 0

#TUNING


model_ft = models.densenet161(pretrained=True)
for param in model_ft.features.parameters():
            param.requires_grad = False
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 5)

#LOAD MODEL PARAMETERS IF NEEDED
model_ft.load_state_dict(torch.load("models/final_fixed_cnn100.pth.tar"))

if use_gpu:
    model_ft = model_ft.cuda()
    print("\nModel loaded in CUDA")
    


optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
solver = train_fixed.Solver(optimizer=optimizer_ft)

model_ft = solver.test_model(model_ft, test_loader, N_test)

for acc in solver.val_acc_history:
    if acc > best_acc:
        best_acc = acc
        
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
plt.savefig('figures/test_final.png', bbox_inches='tight', dpi=200)
plt.show()
        
print("BEST TEST ACCURACY: "+str(best_acc))

acc_history = {'train_loss': solver.train_loss_history,
               'train_acc': solver.train_acc_history,
               'val_loss': solver.val_loss_history,
               'val_acc': solver.val_acc_history}
pickle.dump(acc_history, open('figures/test_final'+'.p','wb') )
#Save the model


