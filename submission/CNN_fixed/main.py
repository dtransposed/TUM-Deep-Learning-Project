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
import DataPreprocessing_fixed as DP

# MAIN CODE

plt.ion()   # interactive mode

use_gpu = torch.cuda.is_available()

# LOAD DATASET
path = '/home/peternagy96/Project/big_dataset/'
data = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=400)



print('orig size:')
print(data['data'].shape)
print('orig size:')
print(data['targets'].shape)

data['data'] = DP.normalize(data['data'])

#split training and validation at random
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

train_loader = torch.utils.data.DataLoader(train1, batch_size=100,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val1, batch_size=100,shuffle=True, num_workers=0)

# Load model and replace last layer
model_ft = models.densenet161(pretrained=True)
for param in model_ft.features.parameters():
            param.requires_grad = False
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 5),
                                     nn.Softmax(dim=1))

#LOAD MODEL PARAMETERS IF NEEDED
#model_ft.load_state_dict(torch.load("models/test.pth.tar"))

if use_gpu:
    model_ft = model_ft.cuda()
    print("\nModel loaded in CUDA")


optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=0.00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
solver = train_fixed.Solver(optimizer=optimizer_ft)

#TRAIN
model_ft = solver.train_model(model_ft, train_loader, val_loader, N_train, N_val, num_epochs=100)

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
plt.savefig('loss_acc_2kfixed.png', bbox_inches='tight', dpi=200)
plt.show()


#Save the model
torch.save(model_ft.state_dict(), "models/2kfixed.pth.tar")
print("\n==========\nModel successfully saved!\n==========")

