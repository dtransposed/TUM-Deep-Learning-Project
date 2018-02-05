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
import VideoDataLoader as VDL
import train
import CNN
import datetime
import pickle

plt.ion()   # interactive mode

vid_cap = 50

start = time.time()
path = '/home/peternagy96/Project/big_dataset/'
data = VDL.load_videos(path, vid_cap, resize_images=False, huge_data=False)

print('orig size:')
print(data['data'].shape)
print('orig size:')
print(data['targets'].shape)

#get date
DT = datetime.datetime.now()
date = str(DT.strftime("%B")[0:3]) + '_' + str(DT.day) + '_'  + str(DT.hour) + ':'  + str(DT.minute)
print(date)

#pickle.dump(data, open('datasets/data'+str(vid_cap)+'_'+date+'.p','wb') )
pickle.dump(data, open('datasets/test'+str(vid_cap)+'_'+date+'.p','wb') )
end = time.time() - start
print("\n==========\nDATA SUCCESSFULLY PICKLED")
print('That took {:.0f}m {:.0f}s'.format(
            end // 60, end % 60),"\n==========")