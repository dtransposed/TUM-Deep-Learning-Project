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

import pytube
import pandas as pd
import os
#type in the class
#cls=Class1
path=os.path.join(os.path.expanduser('~'), 'Project', 'server','Links.csv') #excel table
path2=os.path.join(os.path.expanduser('~'), 'Project', 'server','videos') #save video
ytlist=pd.read_csv(path,header= None)
ytlist=ytlist.iloc[16:17,:]
xname=0
for i in ytlist[2]:
    try:
        xname+=1
        print("1")
        yt = pytube.YouTube(i)
        print(yt)
        #videoname=str(cls+
        #yt.set_filename(videoname)
        #print("25")
        vids=yt.streams.all()
        yt.streams.filter(res="240").all()
        print("3")
        name="file" + str(xname)
        #vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown", filename=name)
        vids[1].download(path2)
        print("4")
        #yt.register_on_complete_callback(frames())
        print("5")
        '''get frames'''

    except:
        pass 
    
import pytube
import pandas as pd

ytlist=pd.read_csv("/Project/server/Links.csv",header= None)
ytlist=ytlist.iloc[2:3,:]
    
    
    


path=os.path.join(os.path.expanduser('~'),'videos') 
directory = os.fsencode(path)
list_files=[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    list_files.append(filename)
    clip = VideoFileClip(filename)
    duration=clip.duration 
    remov_beg=int(duration*param1)
    duration_without_intro=duration-remov_beg
    while remov_beg<duration_without_intro:
        print(remov_beg)
        i=1
        newvid=ffmpeg_extract_subclip(filename, remov_beg, remov_beg+param2)
        i=i+1
        remov_beg=remov_beg+10
        print(remov_beg)

    
    
    
    
    
    