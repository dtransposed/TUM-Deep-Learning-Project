import cv2
import numpy as np
import os

#image folder of the whole class
image_folder1='/home/peternagy96/Project/big_dataset/01_Class_2'
#image folder of of our OpticalFlow data
image_folder2='/home/peternagy96/Project/big_dataset/01_Class_2_OptFlow'

dir1=os.listdir(image_folder1)
dir2=os.listdir(image_folder2)

for folder in dir1:
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
