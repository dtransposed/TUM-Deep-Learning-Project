#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
import math
import VideoPooling as VP

class flowCNN(nn.Module):
    
    def __init__(self, num_classes=3, input_channel=10):
        super(flowCNN, self).__init__()
        self.classifier = nn.Linear(1024, num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 96, kernel_size=7, padding=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()
            
            
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = VP.average_pooling(out)
        out = self.classifier(out)
        if len(out.cpu().data.numpy().shape) == 1:
            out = out.view(1,out.shape[0])
        out = self.softmax(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()