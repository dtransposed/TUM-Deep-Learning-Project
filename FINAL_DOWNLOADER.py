#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:31:53 2018

@author: peternagy96
"""
import pandas as pd
import pytube
import os

ytlist=pd.read_csv('/home/peternagy96/Project/server/01_Class_2/Links.csv',header= None)
ytlist=ytlist.iloc[19:20,:]
xname=0
for i in ytlist[2]:
    try:
        xname+=1
        print("1")
        yt = pytube.YouTube(i)
        print("2")
        #yt.set_filename('Dancing')
        print("2.5")
        vids = yt.streams.all()
        yt.streams.filter(res="240").all()
        print("3")
        name= "file" + str(xname)
        #vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown", filename=name)
        vids[1].download('/home/peternagy96/Project/server/01_Class_2', filename=name)
        print("4")
        #yt.register_on_complete_callback(frames())
        print("5")
    except:
        pass   
    
    
    