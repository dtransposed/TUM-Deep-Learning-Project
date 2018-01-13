#Howdy! Let's get started. First we need some libs
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2
    
class DataImport(object):
    #--1 SPECIFY PARAMETERS
    # =============================================================================
    # imageDir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/all data' #directory to one folder where all training images are
    # imageDir1 = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset/all data' #directory to one folder where all validation images are
    # data_dir = '/home/peter/Desktop/Robotics/DL_Project/models/vgg/Dataset'#this is actual directory of our dataset
    # =============================================================================
    
    #--2 Data normalization--> unit std and zero mean
    #I have defined two functions, to get mean and standard deviation. You need opencv in order to run them
    def get_mean(imageDir):
        image_path_list=[]
        for file in os.listdir(imageDir):
            extension = os.path.splitext(file)[1]
            image_path_list.append(os.path.join(imageDir, file))
        total=0
        for imagePath in image_path_list:
            image = cv2.imread(imagePath)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            total=total+avg_color
        total=total/len(image_path_list)
        return total/255
    
    def get_std(imageDir):
        image_path_list=[]
        for file in os.listdir(imageDir):
            extension = os.path.splitext(file)[1]
            image_path_list.append(os.path.join(imageDir, file))
        list_std=[]
        for imagePath in image_path_list:
            image = cv2.imread(imagePath)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            list_std.append(avg_color)
        list_std=np.asarray(list_std)
        total=np.ndarray.std(list_std,axis=0)
        return total/255
    
    #REMEMBER--> OPENCV RETURNS COLORS IN BRG ORDER. I think for the input in PyTorch 
    # =============================================================================
    # #this does not play any significant role, but beware and keep it in mind
    # train_mean=get_mean(imageDir)
    # val_mean=get_mean(imageDir1)
    # train_std=get_std(imageDir)
    # val_std=get_std(imageDir1)
    # =============================================================================
    
    #--3 Preprocessing
    
    # =============================================================================
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224), #applies crop augumentation, arg is the size of the output
    #         transforms.RandomVerticalFlip(),#applies flip augumentation
    #         transforms.ToTensor(),#converts image to torch.FloatTensor
    #         #normalize according to RGB mean and standard deviation. args are list of 3 arguments, 
    #         #one for every colour
    #         transforms.Normalize(train_mean,train_std)
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(val_mean,val_std)
    #     ]),
    # }
    # =============================================================================
    
    # =============================================================================
    # #function ImageFolder is pytorch's data reader
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    #                   for x in ['train', 'val']}
    # #function DataLoader is pytorch's data loader
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
    #                                              shuffle=True)
    #               for x in ['train', 'val']}
    # 
    # class_names = image_datasets['train'].classes
    # =============================================================================
    
    #--4 Visualisation
    
    def imshow(inp, train_mean, train_std):#, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = train_mean
        std = train_std
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    # =============================================================================
    # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    # =============================================================================
