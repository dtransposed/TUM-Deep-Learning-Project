'''! party07 !'''

import torch
import torch.nn as nn
import torchvision

class Fully_Conv_Block(nn.Module):
    # Gets a pretrained CNN (Currently VGG) and cuts off the FC head
    # Idea: experiment with 'cuts' of the original VGG (e.g. cut of last pooling layer)
        
    def __init__(self, Network_Name):
        
        super(Fully_Conv_Block, self).__init__()
        
        #Takes only vgg11 as intput so far
        if Network_Name == 'vgg11':
            self.model = nn.Sequential(torchvision.models.vgg11(pretrained=True).features)
        
        #Takes only vgg11 as intput so far
        if Network_Name == 'vgg11_bn':
            self.model = nn.Sequential(torchvision.models.vgg11_bn(pretrained=True).features)
        
        if Network_Name == 'densenet121':
            self.model = nn.Sequential(torchvision.models.densenet121(pretrained=True).features)
            
        
    def forward(self, data):
        # takes data of dimension (#frames, height, width, rgb)
        
        data = self.model(data)
        return data
        
        
        
        