from moviepy.editor import *
import imageio
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip
import os
import cv2
import math
from pathlib import Path
from PIL import Image
import simplejson

####SUBCLIP EXTRACTOR#####

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



param1=0.1 #which percentage of the initial video is being removed (stupid talking heads)
param2=6 #how long is the video sequence, with 6fps we get 36 frames


directory = os.fsencode('F:/Wszystko Python/Pytorch Project/test')
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
        
        
##FRAME EXTRACTOR####

directory2 = os.fsencode('F:/Wszystko Python/Pytorch Project/test')
def extract_frames(movie, times, imgdir):
    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, '{}.jpg'.format(t))
        clip.save_frame(imgpath, t)
times=[]
for a in range(36):
    times.append(a)
i=0
for file in os.listdir(directory2):
    print(file)
    filename = os.fsdecode(file)
    os.makedirs('F:/Wszystko Python/Pytorch Project/test/new%s'%(i), exist_ok=True)
    imgdir = 'F:/Wszystko Python/Pytorch Project/test/new%s'%(i)
    extract_frames(filename, times, imgdir)
    i=i+1
