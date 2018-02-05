#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class FLOWNETWORK:
    
    def __init__(self, model_rgb, model_flow):
        self.model_rgb = model_rgb
        self.model_flow = model_flow
        print('test')
        
    def forward(self, data):      
        x_rgb = self.model_rgb.forward(data['data'])       
        x_flow = self.model_flow.forward(data['optflow'])
        print('test')
        return x_rgb, x_flow
        