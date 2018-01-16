#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:34:15 2018

@author: peternagy96
"""

import os
import shutil
import pathlib as pth
  
dir="C:/Users/tatar/Desktop/gggg"
directory = os.listdir("C:/Users/tatar/Desktop/gggg")
for file in directory:
    print(file)
    if (int(file) % 2 == 1):
        hhh= dir + "/" + str(file)
        shutil.rmtree(hhh)
    else:
        continue
