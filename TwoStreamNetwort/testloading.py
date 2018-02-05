#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:48:12 2018

@author: peternagy96
"""

import VideoDataLoader1 as VDL
from PIL import Image
import numpy as np

path = '/home/peternagy96/Project/big_dataset/'

train,val = VDL.split_path(path,20,0.2)

#print(val[0][1][1])
#print(len(val))
#print(np.array(Image.open(val[0][1][0])).shape)

loader = VDL.load_videos_path(val,batchsize=2)

b = loader.__next__()

b.__next__()

print(b.__next__()[0].shape)
