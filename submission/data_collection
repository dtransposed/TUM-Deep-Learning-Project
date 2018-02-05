# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:47:52 2018

@author: tatar
"""
#============================================================
#============================================================
# Downloading Youtube videos
#============================================================
#============================================================
import pandas as pd
import pytube
import os

def video_downloader(path_links, path_save, resolution):
    ytlist = pd.read_csv(path_links,header= None)
    ytlist = ytlist.iloc[0:60,:]
    xname=0
    for i in ytlist[2]:
        try:
            xname+=1
            yt = pytube.YouTube(i)
            vids = yt.streams.all()
            yt.streams.filter(res = resolution).all()
            name= "file" + str(xname)
            vids[1].download(path_save, filename=name)
        except:
            pass  


resolution = '"240"' # 360, 480 
path_save = '/home/peternagy96/Project/server/01_Class_2'
path_links = '/home/peternagy96/Project/server/01_Class_2/Links.csv'
video_downloader(path_links, path_save, resolution)

#============================================================
#============================================================
# Frame extractor
#============================================================
#============================================================
import pandas as pd
import pytube
import cv2
import math
import os

def frame_extractor(frameRate, frameNumber, class_folder):
    xname=0
    os.makedirs(str(xname))
    filelist_class_folder = os.listdir(class_folder)
    for video in filelist_class_folder:
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


frameRate = 6    # frame rate
frameNumber = 36 # number of frames in a folder
class_folder = "/home/peternagy96/Project/server/videos/FPS_0"
frame_extractor(frameRate, frameNumber, class_folder)

#============================================================
#============================================================
# Data trimmer
#============================================================
#============================================================
import os
import shutil
import pathlib as pth
  
def trim_data(folder_path, folder_ind):
    dir=folder_path
    directory = os.listdir(folder_path)
    for file in directory:
        print(file)
        if (int(file) % folder_ind == 1):
            name = dir + "/" + str(file)
            shutil.rmtree(name)
        else:
            continue

    
folder_path = "C:/Users..."      # folders path
folder_ind = 2      # every i th folder will be deleted
trim_data(folder_path, folder_ind)

#============================================================
#============================================================
# Optical flow creator
#============================================================
#============================================================

import cv2
import numpy as np
import os

def optflow_create(image_folder1, image_folder2):
    dir1=os.listdir(image_folder1)
    dir2=os.listdir(image_folder2)
    for folder in dir1:
        try:
            print(folder)
            os.chdir(image_folder1+'/'+folder)
            os.makedirs(image_folder2+'/'+str(folder))
            os.makedirs(image_folder2+'/'+str(folder)+'/'+'h')
            os.makedirs(image_folder2+'/'+str(folder)+'/'+'v') 
            for element in os.listdir(image_folder1+'/'+folder):
                images = [img for img in os.listdir(image_folder1+'/'+folder) if img.endswith(".jpg")]
            for i in range(0,35):
                prev_img = cv2.imread(images[i],flags=cv2.IMREAD_GRAYSCALE)
                next_img = cv2.imread(images[i+1],flags=cv2.IMREAD_GRAYSCALE)
                flow = cv2.calcOpticalFlowFarneback(prev_img,next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
                horz = horz.astype('uint8')
                vert = vert.astype('uint8')
                cv2.imwrite(image_folder2+'/'+str(folder)+'/'+'h'+'/'+'horiz%i.jpg'%(i),horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(image_folder2+'/'+str(folder)+'/'+'v'+'/'+'vert%i.jpg'%(i),vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except:
            pass
image_folder1 = '/home/peternagy96/Project/big_dataset/01_Class_2' #image folder of the whole class
image_folder2 = '/home/peternagy96/Project/big_dataset/01_Class_2_OptFlow' #image folder of of our OpticalFlow data
optflow_create(image_folder1, image_folder2)





