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
    data1 = []
    targets1 = []
    frames1 = []
    data2 = []
    targets2 = []
    frames2 = []
    for i in range(0,N):
        if i in idx_val:
            data2.append(data[i])
            targets2.append(targets[i])
            frames2.append(frames[i])
        else:
            data1.append(data[i])
            targets1.append(targets[i])
            frames1.append(frames[i])
           
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
    N = data['targets'].shape[0]
    for i in range(0,N):
        inputs = torch.from_numpy(np.array(DP.normalize(data['data'][i])))
        targets = np.array([data['targets'][i]])
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        #yield np.ndarray(data['data'][index:index+data['video_frames'][i]]), np.ndarray(data['targets'][i])
        yield inputs, targets
        
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

def load_videos(path,resize_images=False,huge_data=False,vid_cap=10,load_opt_flow=False):
    # This function loads the data which is given in folder structure into a torch.FloatTensor .
    # Optionally the images are resized to a fourth of the original size
    # It returns a dictionary with the data, targets and number frames for each video.
    
    classes = np.array([])
    data = np.array([])
    targets = np.array([])
    video_frames = np.array([])
    dict = None
    
    i=0
    for class_dir in os.listdir(path):
    print(class_dir)    
        if True:
            current_class_dir = path + '/' + class_dir
            i=0
            k = 0
            l=0
            
            data = np.array([])
            targets = np.array([])
            video_frames = np.array([])
            
            if class_dir[-7:] != 'OptFlow':
            
                print(class_dir)
            
                for ren_dir in os.listdir(current_class_dir):
                    
                    print(ren_dir)
                    
                    if k == vid_cap:
                        break
                    
                    k += 1
                    
                    if huge_data:
                        i=0
                        classes = np.array([])
                        data = np.array([])
                        targets = np.array([])
                        video_frames = np.array([])
                    
                    current_ren_dir = current_class_dir + '/' + ren_dir
                    
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
                            
                        if i == 0:
                            data = np.array([vid_np])
                        else:
                            data = np.concatenate((data,[vid_np]),axis=0)
                        
                        video_frames = np.append(video_frames,vid_np.shape[0]).astype(int)
                        
                        i+=1
                    
                    if load_opt_flow:
                        
                        current_flow_dir = current_class_dir + '_OptFlow' + '/' + ren_dir
                        
                        for flow_dir in os.listdir(current_flow_dir):
                            
                            if huge_data:
                                i=0
                                classes = np.array([])
                                data = np.array([])
                                targets = np.array([])
                                video_frames = np.array([])
                            
                            current_ren_dir = current_flow_dir + '/' + flow_dir
                            
                            current_video_dir = current_ren_dir + '/'
                            
                            relative_video_dir = current_video_dir.replace(os.getcwd()+'/',"")
                            
                            if '[' in relative_video_dir or ']' in relative_video_dir:
                                relative_video_dir_new = relative_video_dir.replace('[', '')
                                relative_video_dir_new = relative_video_dir.replace(']', '')
                        
                            img_list = glob.glob(relative_video_dir + '*.jpg')
                        
                            if len(img_list) > 0:
                                
                                img_shape = np.array(Image.open(img_list[0])).shape
                                
                                if resize_images:
                                    vid_np_flow = np.array([resize(np.array(Image.open(img)), (int(img_shape[0]/4),int(img_shape[1]/4),int(img_shape[2])),preserve_range = True) for img in img_list])
                                else:
                                    vid_np_flow = np.array([np.array(Image.open(img)) for img in img_list])
                                    
                                if l == 0:
                                    data_flow = np.array([vid_np_flow])
                                else:
                                    data_flow = np.concatenate((data_flow,[vid_np_flow]),axis=0)
                                
                                
                                l+=1
                
                if not load_opt_flow:
                    dict = {
                        'data': data,
                        'targets': targets,
                        'video_frames': video_frames
                        }
                else:
                    dict = {
                        'data': data,
                        'optflow': data_flow,
                        'targets': targets,
                        'video_frames': video_frames
                        }
                
                yield dict
            
            
                    
            if huge_data:
                dict = {
                    'data': data,
                    'targets': targets,
                    'video_frames': video_frames
                    }
                pickle.dump(dict, open('huge_dataset_trunc_' + ren_dir,'wb') )
                del dict
        
        