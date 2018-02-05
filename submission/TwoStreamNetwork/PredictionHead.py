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
    


