#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:55:08 2018

@author: peternagy96
"""

'''
Train the network with precomputed CNN output
-> faster and CNN fixed anyway
'''

import PredictionHead as PH
import VideoSolver as VS
import VideoClassifier as VC
import glob
import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.optim import lr_scheduler

#learning_rate = 0.0005 #densenet
learning_rate=0.001 # best vgg

# Defining hyperparameters
epochs = 1200
#learning_rate_CNN = 0.000001
output_classes = 5
#network = 'densenet121'
network = 'vgg11'
output_dim = 25088 #vgg1
#output_dim = 50176 #densenet
#
hidden_dim = 4096
#data_train = torch.from_numpy(train)

# Load 5000 sample data with 5 classes
#merge
N = 5000
data_train = torch.zeros([4000,output_dim])
data_val = torch.zeros([1000,output_dim])
data_test = torch.zeros([400,output_dim])
targets_train = torch.zeros([4000]).type(torch.LongTensor)
targets_val = torch.zeros([1000]).type(torch.LongTensor)
targets_test = torch.zeros([400]).type(torch.LongTensor)

for i in range(0,5):
    if i==0:
        data_test=pickle.load(open('conv_out_400_'+network+'_test_class'+str(i)+'.p','rb'))
        targets_test=torch.from_numpy(pickle.load(open('targets_400_'+network+'_test_class'+str(i)+'.p','rb')))
    
    else:
        data_test=torch.cat((data_test, pickle.load(open('conv_out_400_'+network+'_test_class'+str(i)+'.p','rb'))), 0)
        targets_test=torch.cat((targets_test, torch.from_numpy(pickle.load(open('targets_400_'+network+'_test_class'+str(i)+'.p','rb')))), 0)
        
for i in range(0,output_classes):
    print('Load class ' + str(i))
    data_train[i*800:(i+1)*800] =  pickle.load(open('conv_out_5000_'+network+'_train_class'+str(i)+'.p','rb'))
    data_val[i*200:(i+1)*200] = pickle.load(open('conv_out_5000_'+network+'_val_class'+str(i)+'.p','rb'))
    targets_train[i*800:(i+1)*800] = torch.from_numpy(pickle.load(open('targets_5000_train_class'+str(i)+'.p','rb')))
    targets_val[i*200:(i+1)*200] = torch.from_numpy(pickle.load(open('targets_5000_val_class'+str(i)+'.p','rb')))
    
N = targets_train.shape[0] + targets_val.shape[0]

##### Create data loaders
dataset_train = TensorDataset(data_train, targets_train)
dataset_val = TensorDataset(data_val, targets_val)
dataset_test = TensorDataset(data_test, targets_test)
train_loader = torch.utils.data.DataLoader(dataset_train,num_workers=1,batch_size=100)
val_loader = torch.utils.data.DataLoader(dataset_val,num_workers=1,batch_size=100)
test_loader = torch.utils.data.DataLoader(dataset_test,num_workers=1,batch_size=1)

##### Classifier
dim = [10,10,10,20,2]  
FC_param = np.array([[output_dim,hidden_dim],[hidden_dim,hidden_dim],[hidden_dim,output_classes]]) 
#bias_def = True

pred_head = PH.ThreeLayerFCN([output_dim,hidden_dim,output_classes])
##### Use Piotrs code

optimizer_Classifier = torch.optim.SGD(pred_head.parameters(), lr=learning_rate, momentum=0.9)
#optimizer_Classifier = torch.optim.Adam(pred_head.parameters(), lr=learning_rate)

solver = VS.VideoSolver(optim_Classifier=optimizer_Classifier)

pred_hist, pred_hist_train, loss_hist, best_perf, best_ep = solver.train_pre_computed_conv_out(pred_head, train_loader, val_loader, N, num_epochs=epochs)

# performance on test set
pred_test_scores = []
for k, (inputs, targets) in enumerate(test_loader):
    inputs = Variable(inputs)
    targets = Variable(targets)
    
    #Check whether cuda is available and utilize if possible
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
    
    outputs = pred_head.forward(inputs)
    
    # get the prediction -> index of maximum of the output
    _, preds = torch.max(outputs,1)
    
    # zero for wrong and 1 for correct classification
    scores = (preds.cpu() == targets.cpu()).data.numpy()
    
    pred_test_scores.append(scores)
    
perf_test = np.mean(pred_test_scores)

print('Performance on test set: ' + str(perf_test))


results = {
        'pred_hist': pred_hist,
        'pred_hist_train': pred_hist_train,
        'loss_hist': loss_hist,
        'perf_test': perf_test,
        'learning_rate': learning_rate,
        'network': network
        }



#pickle.dump(results,open('final_results_'+str(perf_test)+'.p','wb'))
pickle.dump(results,open('Bogunowicz Final VGG: '+ str(learning_rate)+'.p','wb'))

#pickle.dump(results,open('Final dense test learning rate: 5e-05.p','wb'))
