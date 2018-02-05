import os
import glob
from PIL import Image
import numpy as np
from skimage.transform import resize
import pickle
import DataPreprocessing1 as DP
import torch
import h5py

def split_path(path, N, split):
    # Loads path to N videos in path and performs
    # a split into on the data according to the dezimal
    # value 0 <= split <=1
    
    paths1 = []
    paths2 = []
    
    for class_dir in os.listdir(path):
        current_class_dir = path + '/' + class_dir
        
        if class_dir[-7:] != 'OptFlow':
            vid_dirs = os.listdir(current_class_dir)
            val_vids = np.random.choice(vid_dirs,int(N*split),replace=False)
            for vd in val_vids:
                vid_dirs.remove(vd)
            vid_dirs = vid_dirs[0:int(N*(1-split))]
            
            for vid in vid_dirs:
                current_ren_dir = current_class_dir + '/' + vid
                current_video_dir = current_ren_dir + '/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_rbg = glob.glob(relative_video_dir + '*.jpg')
                
                current_ren_dir = current_class_dir + '_OptFlow/' + vid
                current_video_dir = current_ren_dir + '/h/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_flow_h = glob.glob(relative_video_dir + '*.jpg')
                current_video_dir = current_ren_dir + '/v/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_flow_v = glob.glob(relative_video_dir + '*.jpg')
                
                img_list_flow = []
                for i in range(0,len(img_list_flow_h)):
                    img_list_flow.append(current_ren_dir + '/h/horiz' + str(i) + '.jpg' )
                    img_list_flow.append(current_ren_dir + '/v/vert' + str(i) + '.jpg' )
                
                paths1.append([img_list_rbg,img_list_flow])
            
            for vid in val_vids:
                current_ren_dir = current_class_dir + '/' + vid
                current_video_dir = current_ren_dir + '/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_rbg = glob.glob(relative_video_dir + '*.jpg')
                
                current_ren_dir = current_class_dir + '_OptFlow/' + vid
                current_video_dir = current_ren_dir + '/h/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_flow_h = glob.glob(relative_video_dir + '*.jpg')
                current_video_dir = current_ren_dir + '/v/'
                relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                img_list_flow_v = glob.glob(relative_video_dir + '*.jpg')
                
                img_list_flow = []
                for i in range(0,len(img_list_flow_h)):
                    img_list_flow.append(current_ren_dir + '/h/horiz' + str(i) + '.jpg' )
                    img_list_flow.append(current_ren_dir + '/v/vert' + str(i) + '.jpg' )
                
                paths2.append([img_list_rbg,img_list_flow])
    
    return paths1, paths2
 
def load_videos_path(path,batchsize=0):
    # wrapper generator
    while True:
        yield _load_videos_path(path,batchsize)

def _load_videos_path(path,batchsize=0):
    # generator for iterating over batches in path
    # every yield statement corresponds to one batch
    N = len(path)
    for i in range(0,int(N/batchsize)):
        idx = np.random.choice(np.arange(0,N),batchsize,replace=False)
        #print(idx.shape)
        k=0
        for j in idx:
            #print(j)
            vid = path[j]
            vid_rgb = np.array([np.array(Image.open(img)) for img in vid[0]])
            vid_opt = np.array([np.array(Image.open(img)) for img in vid[1]])
            #print(vid[0])
            if len(vid_rgb) > 2 and len(vid_opt) > 2:
                vid_rgb = DP.normalize(vid_rgb)
                vid_opt = DP.normalize(vid_opt,grey_scale=True)
                if k == 0:
                    all_rgb = np.array([vid_rgb])
                    all_opt = np.array([vid_opt])
                    #print(vid[0][0])
                    #print(vid[0][0][47])
                    targets = np.array([int(vid[0][0][40])])
                    #targets = np.array([int(vid[0][0][42])])
                else:
                    all_rgb = np.concatenate((all_rgb,[vid_rgb]),axis=0)
                    all_opt = np.concatenate((all_opt,np.array([vid_opt])),axis=0)
                    targets = np.concatenate((targets,np.array([int(vid[0][0][40])])),axis=0)
                    #targets = np.concatenate((targets,np.array([int(vid[0][0][42])])),axis=0)
                k += 1
        yield all_rgb,all_opt,targets
            

    
    