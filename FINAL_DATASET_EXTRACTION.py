#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:56:33 2018

@author: peternagy96
"""

import pandas as pd
import pytube
import cv2
import math
import os

frameRate = 6    # frame rate
frameNumber = 36 # number of frames in a folder
class_folder = "/home/peternagy96/Project/server/videos/FPS_0"

xname=7061
os.makedirs(str(xname))
filelist_class_folder = os.listdir(class_folder)

for video in filelist_class_folder:
    print(video)
    print('Has been processed') 
    
    cap = cv2.VideoCapture(video)

    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        try:
            frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        
        except:
            pass

        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = str(xname) + "/image_" + str(int(frameId)) + ".jpg"
            cv2.imwrite(filename, frame)
        
            file = str(class_folder) + "/" + str(xname)
            number_of_files = sum(len(files) for _, _, files in os.walk(file))
            if (number_of_files == frameNumber):
                xname+=1
                os.makedirs(str(xname))
    cap.release()

    
        