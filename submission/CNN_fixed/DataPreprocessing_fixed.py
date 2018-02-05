import numpy as np
import torch

def normalize(data):
    # normalizes the data such that it has zero mean and std = 1
    # assumes data of input dimensions [Number images, !!!Color channel!!!, Height, Width]
    # returns normalized data in a torch.FloatTensor
    
    mean = np.mean(data, axis=(0,1,2))
    std = np.std(data, axis=(0,1,2))
    
    data = (data-mean)/std
    ds = data.shape
    data = np.reshape(data, (ds[0],ds[1],ds[2],ds[3]))
    #data = data.astype('float32')
    #data = torch.from_numpy(data).float()
    
    return data