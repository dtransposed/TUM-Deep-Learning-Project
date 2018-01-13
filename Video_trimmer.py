



from moviepy.editor import *
import imageio
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import os


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name,ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = name+ "%sSUB%d_%d.%s"%(name, T1, T2, ext)
    
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
      "-i", filename,
      "-ss", "%0.2f"%t1,
      "-t", "%0.2f"%(t2-t1),
      "-vcodec", "copy", "-acodec", "copy", targetname]
    
    subprocess_call(cmd)
    
ffmpeg_extract_subclip("testvideo.webm", 3, 7, targetname="test.webm")





import pandas as pd
import pytube
import cv2
import math
import os

cap = cv2.VideoCapture("test.webm")
frameRate =25# cap.get(100) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    #frame = cv2.resize(frame, (480, 270)) 
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        #filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
        cv2.imwrite("filename.jpg", frame)
cap.release()
print("6")

