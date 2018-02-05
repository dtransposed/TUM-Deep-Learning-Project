'''! party07 !'''

import torch
import torch.nn as nn

class ThreeLayerFCN(nn.Module):
        
    def __init__(self, dim):
        
        super(ThreeLayerFCN, self).__init__()
        
        self.dim = dim
        
        self.input_layer = nn.Linear(dim[0], dim[1], bias=True)
        
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(dim[1], dim[2], bias=True)
        
        self.softmax = nn.Softmax(dim=1)
        
            
        
    def forward(self, xout):
        
        xout = self.input_layer(xout)
        
        xout = self.relu(xout)
        
        xout = self.output_layer(xout)
        
        if len(xout.cpu().data.numpy().shape) == 1:
            xout = xout.view(1,xout.shape[0])
        
        return self.softmax(xout)
    
class NLayerFCN(nn.Module):
    # Gets a pretrained CNN (Currently VGG) and cuts off the FC head
        
    def __init__(self, dim):
        
        super(NLayerFCN, self).__init__()
        
        self.dim = dim
        
        D = len(dim)
        self.layer = dict()
        self.layer['linear1'] = nn.Linear(dim[0], dim[1], bias=True)
        self.layer['relu1'] = nn.ReLU()
        
        #self.input_layer = nn.Linear(dim[0], dim[1], bias=True)
        #self.relu = nn.ReLU()
        
        for i in range(1,D-2):
            self.layer['linear'+str(i+1)] = nn.Linear(dim[i], dim[i+1], bias=True)
            self.layer['relu'+str(i+1)] = nn.ReLU()
            
        self.output_layer = nn.Linear(dim[D-2], dim[D-1], bias=True)
        
        self.softmax = nn.Softmax(dim=1)
        
            
        
    def forward(self, xout):
        
        #xout = self.input_layer(xout)
        #xout = self.relu(xout)
        
        for i in range(1,len(self.dim)):
            xout = self.layer['linear'+str(i)](xout)
            xout = self.layer['relu'+str(i)](xout)
        
        xout = self.output_layer(xout)
        
        if len(xout.cpu().data.numpy().shape) == 1:
            xout = xout.view(1,xout.shape[0])
        
        return self.softmax(xout)

dim = [10,10,10,20,2]

NN = NLayerFCN(dim)

print(NN)

