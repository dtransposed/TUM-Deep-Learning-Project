import os
import glob
from PIL import Image
import numpy as np
from skimage.transform import resize
import pickle
import DataPreprocessing as DP
import torch

def split_dataset(dataset, size=0.5):
    # randomly splits the dataset into two subsets of size and 1-size
    data = dataset['data']
    targets = dataset['targets']
    frames = dataset['video_frames']
    
    #get random elements
    N = targets.shape[0]
    idx_val = np.random.choice(range(0,N),int(size*N),replace=False)
    
    # perform split
    datalist1 = []
    targets1 = []
    frames1 = []
    datalist2 = []
    targets2 = []
    frames2 = []
    j = 0
    for i in range(0,N):
        if i in idx_val:
            datalist2.append(data[j:j+frames[i]])
            targets2.append(targets[i])
            frames2.append(frames[i])
        else:
            datalist1.append(data[j:j+frames[i]])
            targets1.append(targets[i])
            frames1.append(frames[i])
           
        j+=frames[i]
    
    i = 0
    data1 = []
    for el in datalist1:
        for pic in el:
            data1.append(pic)
    i = 0
    data2 = []
    for el in datalist2:
        for pic in el:
            data2.append(pic)
    
    dict1 = {
        'data' : np.array(data1),
        'targets' : np.array(targets1),
        'video_frames' : np.array(frames1)
        }
    
    dict2 = {
        'data' : np.array(data2),
        'targets' : np.array(targets2),
        'video_frames' : np.array(frames2)
        }    
    
    return dict1, dict2


def iterate_videos(data):
    while True:
        yield _iterate_videos(data)

def _iterate_videos(data):
    # This is a generator function. It iterates over the data
    # and returns the frames of one video at a time with each call.
    # To-Do: implement batchsize
    
    i = 0
    index = 0
    N = data['targets'].shape[0]
    for i in range(0,N):
        inputs = torch.from_numpy(np.array(DP.normalize(data['data'][index:index+data['video_frames'][i]])))
        targets = np.array([data['targets'][i]])
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        #yield np.ndarray(data['data'][index:index+data['video_frames'][i]]), np.ndarray(data['targets'][i])
        yield inputs, targets
        
        #index += data['video_frames'][i]
        #i += 1
        #i = i % N
        #if i==0:
            #index = 0

def iterate_videos_from_pickle(files, normalize=False, use_first_hundred=False):
    while True:
        yield _iterate_videos_from_pickle(files, normalize=normalize, use_first_hundred=use_first_hundred)

def _iterate_videos_from_pickle(files, normalize=False, use_first_hundred=False):
    # This is a generator function. It iterates over the data
    # and returns the frames of one video at a time with each call.
    # To-Do: implement batchsize
    
    for file in files:
            
        data = pickle.load(open(file,'rb'))
        
        index = 0
        N = data['targets'].shape[0]
        
        for i in range(0,N):
            
            if normalize:
                if use_first_hundred:
                    if data['video_frames'][i] > 100:
                        inputs = DP.normalize(data['data'][index:index+100])
                    else: 
                        inputs = DP.normalize(data['data'][index:index+data['video_frames'][i]])
                else:
                    inputs = DP.normalize(data['data'][index:index+data['video_frames'][i]])
            else:
                if use_first_hundred:
                    if data['video_frames'][i] > 100:
                        inputs = torch.from_numpy(np.array(data['data'][index:index+100]))
                    else:
                        inputs = torch.from_numpy(np.array(data['data'][index:index+data['video_frames'][i]]))
                else:
                    inputs = torch.from_numpy(np.array(data['data'][index:index+data['video_frames'][i]]))
            
            targets = np.array([data['targets'][i]])   
            targets = torch.from_numpy(targets).type(torch.LongTensor)
            
            yield inputs, targets
            
            index += data['video_frames'][i]
            i += 1

def load_videos(path,resize_images=False,huge_data=False):
    # This function loads the data which is given in folder structure into a torch.FloatTensor .
    # Optionally the images are resized to a fourth of the original size
    # It returns a dictionary with the data, targets and number frames for each video.
    
    classes = np.array([])
    data = np.array([])
    targets = np.array([])
    video_frames = np.array([])
    
    i=0
    for class_dir in os.listdir(path):
        
        current_class_dir = path + '/' + class_dir
        
        print(current_class_dir)
        
        k = 0 # DELETEEEE
        
        for ren_dir in os.listdir(current_class_dir):
            
            if k == 4: # DELETEEEE
                break # DELETEEEE
            
            print(ren_dir)
            
            if huge_data:
                i=0
                classes = np.array([])
                data = np.array([])
                targets = np.array([])
                video_frames = np.array([])
            
            current_ren_dir = current_class_dir + '/' + ren_dir
            
            #for video_dir in os.listdir(current_ren_dir):
                
            #current_video_dir = current_ren_dir + '/' + video_dir + '/'
            current_video_dir = current_ren_dir + '/'
            relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
            
            if '[' in relative_video_dir or ']' in relative_video_dir:
                relative_video_dir_new = relative_video_dir.replace('[', '')
                relative_video_dir_new = relative_video_dir.replace(']', '')
            
            img_list = glob.glob(relative_video_dir + '*.jpg')
            
            if len(img_list) > 0:
                
                img_shape = np.array(Image.open(img_list[0])).shape
                
                targets = np.append(targets,int(class_dir[9])-1).astype(int)
                
                if resize_images:
                    vid_np = np.array([resize(np.array(Image.open(img)), (int(img_shape[0]/4),int(img_shape[1]/4),int(img_shape[2])),preserve_range = True) for img in img_list])
                else:
                    vid_np = np.array([np.array(Image.open(img)) for img in img_list])
                
                vid_np = DP.normalize(vid_np)
                
                if i == 0:
                    data = vid_np
                else:
                    data = np.concatenate((data,vid_np),axis=0)
                
                video_frames = np.append(video_frames,vid_np.shape[0]).astype(int)
                
                i+=1
                    
            if huge_data:
                dict = {
                    'data': data,
                    'targets': targets,
                    'video_frames': video_frames
                    }
                pickle.dump(dict, open('huge_dataset_trunc_' + ren_dir,'wb') )
                del dict
            
            k+=1 # DELETEEEE
    
    if not huge_data:
        dict = {
            'data': data,
            'targets': targets,
            'video_frames': video_frames
            }
        return dict
    return -1
