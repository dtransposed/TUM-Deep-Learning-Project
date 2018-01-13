# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:34:26 2018

@author: tatar
"""


import pandas as pd
import pytube

def frame_extract()
    
    import pytube
    import cv2
    import math
    import os
    print(file)
    x = os.listdir("C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0")
    
       print(file)
    for filename in os.listdir("C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0"):
        videoFile = filename
        imagesFolder = "C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0"
        cap = cv2.VideoCapture(videoFile)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameRate = 6 #frame rate
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            frame = cv2.resize(frame, (400, 400)) 
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename = "/image_" +  str(int(frameId)) + ".jpg"
                cv2.imwrite(filename, frame)
        cap.release()
      
    return None

#type in the class
cls=Class1
ytlist=pd.read_csv('C:/Users/tatar/Desktop/DL Project/Data_set_3/Links.csv',header= None)
ytlist=ytlist.iloc[2:3,:]
xname=0
for i in ytlist[2]:
    try:
        xname+=1
        print("1")
        yt = pytube.YouTube(i)
        print(yt)
        videoname=str(cls+
        yt.set_filename(videoname)
        print("2.5")
        vids= yt.streams.all()
        yt.streams.filter(res="240").all()
        print("3")
        name="file" + str(xname)
        #vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown", filename=name)
        vids[1].download(r"C:/Users/tatar/Desktop/DL Project/Data_set_3/Class 1")
        print("4")
        #yt.register_on_complete_callback(frames())
        print("5")
        '''get frames'''

    except:
        pass 
    