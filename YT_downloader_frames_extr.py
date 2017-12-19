# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:20:48 2017

@author: tatar
"""

# extracting frames into a proper folder structer based on a defined structure
import pandas as pd
import pytube
import cv2
import math
import os

        
os.chdir("C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS") # path of the folder with video folders
class_folder_path = os.getcwd().replace("\\", "/")
class_folder_strucutre = os.listdir(class_folder_path) 

for folder in class_folder_strucutre:
    video_folder = str(class_folder_path)+ "/" + str(folder) # path of the folder with videos
    filelist_video_folder = os.listdir(video_folder) # list of the folders with videos
    plik=0 # for naming
    video_folder_name = str(folder) + '_frames'
    os.makedirs(video_folder_name) #creates frame folder
     
    for video in filelist_video_folder:
        try:
            plik = plik+1 #just for naming
            print("plik:", plik)
            videoFile = video_folder + "/"+ str(video) #path to find videos in video folder
            videoFile2 = video_folder_name + "/"+ str(video) # path to save frames in frame folder
            FolderName = str(videoFile2) # name of the frame subfolder
            os.makedirs(FolderName)
            
            imagesFolder = FolderName # where to save frames
            cap = cv2.VideoCapture(videoFile)
            frameRate =500# cap.get(100) #frame rate
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
                #frame = cv2.resize(frame, (480, 270)) 
                if (ret != True):
                    break
                if (frameId % math.floor(frameRate) == 0):
                    filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
                    cv2.imwrite(filename, frame)
            cap.release()
            print("6")
        except:
            pass

    

# downloading youtube videos     

ytlist=pd.read_csv('C:/Users/tatar/Desktop/DL Project/book1.csv',header= None)
ytlist=ytlist.iloc[60:100,:]
xname=0
for i in ytlist[2]:
    try:
        xname+=1
        print("1")
        yt = pytube.YouTube(i)
        print("2")
        #yt.set_filename('Dancing')
        print("2.5")
        vids= yt.streams.all()
        print("3")
        name="file" + str(xname)
        #vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown", filename=name)
        vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown")
        print("4")
        #yt.register_on_complete_callback(frames())
        print("5")
        '''get frames'''

    except:
        pass 
    
''' 
import pandas as pd
import pytube
import cv2
import math
import os

#folder = "C:/Users/tatar/Desktop/DL Project/ytdown/"
#for file in os.listdir(folder):
    
print(file)
x = os.listdir("C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0")

   print(file)


for filename in os.listdir("C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0"):
    videoFile = filename
    
    imagesFolder = "C:/Users/tatar/Desktop/DL Project/Data_set_1/Class_1_FPS/FPS_0"
    cap = cv2.VideoCapture(videoFile)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameRate = cap.get(100) #frame rate
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
        
 x   
'''






'''
import pandas as pd
import pytube

ytlist=pd.read_csv('C:/Users/tatar/Desktop/DL Project/ytdown/book1.csv',header= None)
ytlist=ytlist.iloc[0:8,:]

for i in ytlist:
    try:
        yt = pytube.YouTube(i)
        vids= yt.streams.all()
        yt.streams.filter(res="360").filter(subtype='mp4').all()
        for i in range(len(vids)):
            print(i,'. ',vids[i])
        print('done' + str(i))
    except:
        pass
vnum = int(input("Enter vid num: "))
vids[vnum].download(r"C:/Users/tatar/Desktop/DL Project/ytdown")
'''