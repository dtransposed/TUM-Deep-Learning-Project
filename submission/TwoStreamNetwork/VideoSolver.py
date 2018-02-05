'''
Created on 12 Jan 2018

@author: Janis
'''

import numpy as np
import torch.optim
import torch.nn
import torch.cuda
import torch
from torch.autograd import Variable

class VideoSolver:
    
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    
    def __init__(self, optim=[torch.optim.Adam,torch.optim.Adam], optim_args={}, loss_func=torch.nn.CrossEntropyLoss()):
        self.optim_spatial = optim[0]
        self.optim_temporal = optim[1]
        self.loss_func = loss_func
        
    def train(self, model, train_loader, val_loader, N, num_epochs=10, train_CNN=False, train_rgb=True, train_opt=True, pure_train=False):
        # trains video classification model consisting
        # of two flows : spatial / temporal analysis
        
        model_spatial, model_temporal = model
        
        if torch.cuda.is_available():
            model_temporal.cuda()
            model_spatial.cuda()
            
        # store the history of the learning process
        loss_history = []
        pred_scores_history = []
        loss_history_opt = []
        pred_scores_history_train = []
        best_perf = 0.0
        best_ep = 0
        
        #iterate over epochs
        for i in range(1,num_epochs+1):
            
            print('### EPOCH ' + str(i) + ' ###')
            run_loss = 0     
            
            #iterate over batches
            
            ################## TRAINING #####################
            for j, (inputs_rgb, inputs_opt, targets) in enumerate(train_loader.__next__()):
                
                
                targets = torch.from_numpy(targets)
                targets = targets = Variable(targets)
                if torch.cuda.is_available():
                    targets = targets.cuda()
                if train_rgb:
                    i = 0
                    self.optim_spatial.zero_grad()
                    for inp in inputs_rgb:
                        
                        inputs = torch.from_numpy(np.array(inp))
                        
                        inputs = Variable(inputs)
                        
                        #Check whether cuda is available and utilize if possible
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                        outputs = model_spatial.forward(inputs)
                        
                        if len(outputs.cpu().data.numpy().shape) == 1:
                            outputs = outputs.view(1,outputs.shape[0])
                            
                        loss = self.loss_func(outputs, targets[i])
                        
                        loss.backward()
                
                        loss_history.append(loss.cpu().data.numpy()[0])
                        run_loss += loss.cpu().data.numpy()[0]
                        i+=1
                        
                    self.optim_spatial.step()
                
                if train_opt:
                    i = 0
                    self.optim_temporal.zero_grad()
                    for inp in inputs_opt:
                        inputs = torch.from_numpy(inp)
                        inputs = Variable(inputs)
                        
                        #Check whether cuda is available and utilize if possible
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                        
                        #change view
                        inputs = inputs.view(7,10,224,224).type(torch.cuda.FloatTensor)
                        
                        outputs = model_temporal.forward(inputs)
                        
                        if len(outputs.cpu().data.numpy().shape) == 1:
                            outputs = outputs.view(1,outputs.shape[0])
                        
                        loss = self.loss_func(outputs, targets[i])
                        
                        loss.backward()
                
                        loss_history_opt.append(loss.cpu().data.numpy()[0])
                        run_loss += loss.cpu().data.numpy()[0]
                        i+=1
                    self.optim_temporal.step() 
            
            if not pure_train:
                #################### TRAINING ACCURACY #####################
                pred_scores_rgb_train = []
                pred_scores_opt_train = []
                pred_targ = []
                for j, (inputs_rgb, inputs_opt, targets) in enumerate(train_loader.__next__()):
                    
                    targets = torch.from_numpy(targets)
                    targets = targets = Variable(targets)
                    if torch.cuda.is_available():
                        targets = targets.cuda()
                    if train_rgb:
                        i = 0
                        for inp in inputs_rgb:
                            inputs = torch.from_numpy(np.array(inp))
                            #targets = torch.from_numpy(targets[i])
                            inputs = Variable(inputs)
                            #targets = Variable(targets)
                            
                            #Check whether cuda is available and utilize if possible
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                            
                            outputs = model_spatial.forward(inputs)
                            
                            if len(outputs.cpu().data.numpy().shape) == 1:
                                outputs = outputs.view(1,outputs.shape[0])
        
                            # get the prediction -> index of maximum of the output
                            _, preds = torch.max(outputs,1)
                            
                            # zero for wrong and 1 for correct classification
                            scores = (preds.cpu() == targets[i].cpu()).data.numpy()
                            
                            pred_scores_rgb_train.append(scores)
                            
                            i+=1
                    
                    if train_opt:
                        i = 0
                        for inp in inputs_opt:
                            inputs = torch.from_numpy(inp)
                            #targets = torch.from_numpy(np.array(targets[i]))
                            
                            inputs = Variable(inputs)
                            #targets = Variable(targets)
                            
                            #Check whether cuda is available and utilize if possible
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                            
                            #change view
                            inputs = inputs.view(7,10,224,224).type(torch.cuda.FloatTensor)
                            
                            outputs = model_temporal.forward(inputs)
                            
                            if len(outputs.cpu().data.numpy().shape) == 1:
                                outputs = outputs.view(1,outputs.shape[0])
                            
                            # get the prediction -> index of maximum of the output
                            _, preds = torch.max(outputs,1)
                    
                            # zero for wrong and 1 for correct classification
                            scores = (preds.cpu() == targets[i].cpu()).data.numpy()
                    
                            pred_scores_opt_train.append(scores)
                            
                            i+=1
                        
                ############## VALIDATION ACCURACY ################
                # iterate over validation set and and
                pred_scores_rgb = []
                pred_scores_opt = []
                for k, (inputs_rgb, inputs_opt,  targets) in enumerate(val_loader.__next__()):
                    
                    targets = torch.from_numpy(np.array(targets))
                    
                    targets = Variable(targets)
            
                    if train_rgb:
                        inputs = torch.from_numpy(inputs_rgb)
                        inputs = inputs.view(inputs.shape[1:])
                        inputs = Variable(inputs)
                        
                        #Check whether cuda is available and utilize if possible
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                
                        #outputs = video_model.forward(inputs)
                        outputs = model_spatial.forward(inputs)
                        #outputs = outputs.mean(0)
                        
                        if len(outputs.cpu().data.numpy().shape) == 1:
                            outputs = outputs.view(1,outputs.shape[0])
                        
                        
                        # get the prediction -> index of maximum of the output
                        _, preds = torch.max(outputs,1)
                
                        # zero for wrong and 1 for correct classification
                        scores = (preds.cpu() == targets.cpu()).data.numpy()
                
                        pred_scores_rgb.append(scores)
                    
                    if train_opt:
                        inputs = torch.from_numpy(inputs_opt)
                        #targets = torch.from_numpy(targets)
                        inputs = Variable(inputs)
                        #targets = Variable(targets)
                
                        #Check whether cuda is available and utilize if possible
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                            #targets = targets.cuda()
                
                        #outputs = video_model.forward(inputs)
                        inputs = inputs.view(7,10,224,224).type(torch.cuda.FloatTensor)
                        
                        outputs = model_temporal.forward(inputs)
                        
                        if len(outputs.cpu().data.numpy().shape) == 1:
                            outputs = outputs.view(1,outputs.shape[0])
                        #outputs = outputs.mean(0)
                        
                        
                        # get the prediction -> index of maximum of the output
                        _, preds = torch.max(outputs,1)
                
                        # zero for wrong and 1 for correct classification
                        scores = (preds.cpu() == targets.cpu()).data.numpy()
                
                        pred_scores_opt.append(scores)
                
                mean_perf_opt = 0
                mean_perf_opt_train = 0
                if train_opt:
                    # store the accuracy for each epoch
                    mean_perf_opt = np.mean(pred_scores_opt)
                    # store the accuracy for each epoch
                    mean_perf_opt_train = np.mean(pred_scores_opt_train)
            
                mean_perf_rgb = 0
                mean_perf_rgb_train = 0
                if train_rgb:
                    # store the accuracy for each epoch
                    mean_perf_rgb = np.mean(pred_scores_rgb)
                    # store the accuracy for each epoch
                    mean_perf_rgb_train = np.mean(pred_scores_rgb_train)
                
                #pred_scores_history.append(mean_perf)
                
                if train_rgb:
                    print('SPATIAL PERF ON VAL: ' + str(mean_perf_rgb))
                    print('SPATIAL PERF ON TRAIN: ' + str(mean_perf_rgb_train))
                
                if train_opt:
                    print('TEMPORAL PERF ON VAL: ' + str(mean_perf_opt))
                    print('TEMPORAL PERF ON TRAIN: ' + str(mean_perf_opt_train))
                
                pred_scores_history.append([mean_perf_rgb,mean_perf_opt])
                
                pred_scores_history_train.append([mean_perf_rgb_train,mean_perf_opt_train])
            
        return pred_scores_history, pred_scores_history_train, loss_history, best_perf, best_ep
    
    