#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:23:30 2018

@author: peternagy96
"""

import VideoDataLoader as VDL
import pickle

#path = '/home/peternagy96/Downloads/Frames_for _Damian' 
path = '/home/peternagy96/Project/big_dataset'

#data = VDL.load_videos(path, resize_images=False, huge_data=False)
data_loader = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=10, load_opt_flow=True)

for i, data in enumerate(data_loader):
    print(data['data'].shape)
    print(data['optflow'].shape)