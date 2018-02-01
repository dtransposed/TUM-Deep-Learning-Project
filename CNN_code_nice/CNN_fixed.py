#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
import math

class CNN(nn.Module):
    
    def __init__(self, model, num_classes=3):
        super(CNN, self).__init__()
        self.classifier = nn.Linear(1024, num_classes)
        self.features = nn.Sequential(models.densenet161(pretrained=True))
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
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