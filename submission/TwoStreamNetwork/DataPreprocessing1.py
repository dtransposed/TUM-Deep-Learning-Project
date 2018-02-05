import numpy as np
import torch

def normalize(data, grey_scale=False):
    # normalizes the data such that it has zero mean
    # assumes data of input dimensions [Number images, !!!Color channel!!!, Height, Width]
    # returns normalized data in a numpy array
    
    mean = np.mean(data, axis=(0,1,2))
    std = np.std(data, axis=(0,1,2))
    data = (data-mean)
    ds = data.shape
    if not grey_scale:
        data = np.reshape(data, (ds[0],ds[3],ds[1],ds[2]))
    data = data.astype('float32')
    
    return data