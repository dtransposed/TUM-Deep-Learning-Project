#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:24:28 2018

@author: peternagy96
"""

import VideoDataLoader as VDL
import numpy as np
import pickle

#path = '/home/peternagy96/Downloads/Frames_for _Damian' 
path = '/home/peternagy96/Project/big_dataset'

#data = VDL.load_videos(path, resize_images=False, huge_data=False)
#data = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=100)

#print(data['data'].shape)

results = pickle.load(open('final_results_0.034.p','rb'))

print(np.array(results['learning_rates']).repeat(len(results['loss_hist'])/len(results['learning_rates'])))
print(results['loss_hist'])
print(len(results['loss_hist'])/len(results['learning_rates']))

