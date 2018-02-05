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
from VideoClassifier import VideoClassifier

class VideoSolver:
    
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    
    def __init__(self, optim_CNN=torch.optim.Adam, optim_Classifier=torch.optim.Adam, optim_args={}, loss_func=torch.nn.CrossEntropyLoss()):
        self.optim_CNN = optim_CNN
        self.optim_Classifier = optim_Classifier
        self.loss_func = loss_func
        
    def train(self, video_model: VideoClassifier, train_loader, val_loader, N, num_epochs=10, train_CNN=False):
        
        if torch.cuda.is_available():
            video_model.CNN_Model.cuda()
            video_model.Classifier.cuda()
        # store the history of the learning process
        loss_history = []
        pred_scores_history = []
        best_perf = 0.0
        best_ep = 0
        
        if not train_CNN:
            conv_output_videos = torch.zeros(N, video_model.Classifier.dim[0])
        
        #iterate over epochs
        for i in range(1,num_epochs+1):
            
            print('### EPOCH ' + str(i) + ' ###')
                  
            #iterate over batches
            for j, (inputs, targets) in enumerate(train_loader.__next__()):
                
                #cast to variable
                if train_CNN:
                    inputs = Variable(inputs)
                else:
                    if i == 1:
                        inputs = Variable(inputs)
                    else:
                        inputs = conv_output_videos[j]
                        inputs = Variable(inputs)
                
                targets = Variable(targets)
                
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                # Zero the gradients of the model parameters
                self.optim_Classifier.zero_grad()
        
                # If we train the CNN model as well zero grads
                if train_CNN:
                    self.optim_CNN.zero_grad() # Zero the gradients of the model parameters
        
                #self.optim_Classifier.zero_grad() # Zero the gradients of the model parameters
        
                # Compute the output of the model
                if train_CNN:
                    outputs = video_model.forward(inputs)
                if not train_CNN:
                    if i == 1:
                        outputs = video_model.CNN_Model(inputs)
                        outputs = video_model.pooling(outputs)
                        conv_output_videos[j] = outputs.cpu().data
                        outputs = video_model.Classifier(outputs)
                    else:
                        outputs = video_model.Classifier(inputs)
                
                print('outputs')
                print(outputs)
                print('targets')
                print(targets)
                
                loss = self.loss_func(outputs, targets)
                
                loss.backward()
                
                self.optim_Classifier.step()
                
#                 if train_CNN:
#                     # Used loss function
#                     loss_func=torch.nn.CrossEntropyLoss()
#                     # Compute loss with respect to the targets
#                     loss = loss_func(outputs,targets)
#                     # compute gradients of parameters
#                     loss.backward()
#                     # update parameters
#                     self.optim_CNN.step()
        
                loss_history.append(loss.cpu().data.numpy()[0])
            
            
            # iterate over validation set and and
            pred_scores = []
            for k, (inputs, targets) in enumerate(val_loader.__next__()):
                if train_CNN:
                    inputs = Variable(inputs)
                else:
                    inputs = conv_output_videos[k]
                    inputs = Variable(inputs)
                targets = Variable(targets)
        
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                #outputs = video_model.forward(inputs)
                if train_CNN:
                    outputs = video_model.forward(inputs)
                if not train_CNN:
                    outputs = video_model.Classifier(inputs)
                    
                # get the prediction -> index of maximum of the output
                _, preds = torch.max(outputs,1)
        
                # zero for wrong and 1 for correct classification
                scores = (preds.cpu() == targets.cpu()).data.numpy()
        
                pred_scores.append(scores)
        
            # store the accuracy for each epoch
            mean_perf = np.mean(pred_scores)
            
            if mean_perf > best_perf:
                best_perf = mean_perf 
                best_ep = i
            
            pred_scores_history.append(mean_perf)
            
            pred_train_scores = []
            for k, (inputs, targets) in enumerate(train_loader.__next__()):
                #inputs = Variable(inputs)
                if train_CNN:
                    inputs = Variable(inputs)
                else:
                    inputs = conv_output_videos[k]
                    inputs = Variable(inputs)
                targets = Variable(targets)
                
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                #outputs = video_model.forward(inputs)
                if train_CNN:
                    outputs = video_model.forward(inputs)
                if not train_CNN:
                    outputs = video_model.Classifier(inputs)
        
                # get the prediction -> index of maximum of the output
                _, preds = torch.max(outputs,1)
        
                # zero for wrong and 1 for correct classification
                scores = (preds.cpu() == targets.cpu()).data.numpy()
        
                pred_train_scores.append(scores)
            print('After Epoch ' + str(i) + ' validation acc / training acc: ' + 
                  str(np.mean(pred_scores)) + ' / ' + str(np.mean(pred_train_scores)))
        
        return pred_scores_history, loss_history, best_perf, best_ep
    
    def train_pre_computed_conv_out(self, video_model: VideoClassifier, train_loader, val_loader, N, num_epochs=10, train_CNN=False):
        
        if torch.cuda.is_available():
            video_model.cuda()
        # store the history of the learning process
        loss_history = []
        pred_scores_history = []
        pred_scores_history_train = []
        best_perf = 0.0
        best_ep = 0
        
        if not train_CNN:
            conv_output_videos = torch.zeros(N, video_model.dim[0])
        
        #iterate over epochs
        for i in range(1,num_epochs+1):
            
            print('### EPOCH ' + str(i) + ' ###')
                  
            #iterate over batches
            for j, (inputs, targets) in enumerate(train_loader):
                
                #cast to variable
                inputs = Variable(inputs)
                targets = Variable(targets)
                
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                # Zero the gradients of the model parameters
                self.optim_Classifier.zero_grad()
        
                # If we train the CNN model as well zero grads
                if train_CNN:
                    self.optim_CNN.zero_grad() # Zero the gradients of the model parameters
        
                #self.optim_Classifier.zero_grad() # Zero the gradients of the model parameters
        
                # Compute the output of the model
                outputs = video_model.forward(inputs)
                
                loss = self.loss_func(outputs, targets)
                
                loss.backward()
                
                self.optim_Classifier.step()
                
#                 if train_CNN:
#                     # Used loss function
#                     loss_func=torch.nn.CrossEntropyLoss()
#                     # Compute loss with respect to the targets
#                     loss = loss_func(outputs,targets)
#                     # compute gradients of parameters
#                     loss.backward()
#                     # update parameters
#                     self.optim_CNN.step()
        
                loss_history.append(loss.cpu().data.numpy()[0])
            
            
            # iterate over validation set and and
            pred_scores = []
            for k, (inputs, targets) in enumerate(val_loader):
                inputs = Variable(inputs)
                targets = Variable(targets)
        
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                outputs = video_model.forward(inputs)
        
                # get the prediction -> index of maximum of the output
                _, preds = torch.max(outputs,1)
        
                # zero for wrong and 1 for correct classification
                scores = (preds.cpu() == targets.cpu()).data.numpy()
        
                pred_scores.append(scores)
        
            # store the accuracy for each epoch
            mean_perf = np.mean(pred_scores)
            
            if mean_perf > best_perf:
                best_perf = mean_perf 
                best_ep = i
            
            pred_scores_history.append(mean_perf)
            
            pred_train_scores = []
            for k, (inputs, targets) in enumerate(train_loader):
                inputs = Variable(inputs)
                targets = Variable(targets)
                
                #Check whether cuda is available and utilize if possible
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
        
                outputs = video_model.forward(inputs)
        
                # get the prediction -> index of maximum of the output
                _, preds = torch.max(outputs,1)
        
                # zero for wrong and 1 for correct classification
                scores = (preds.cpu() == targets.cpu()).data.numpy()
        
                pred_train_scores.append(scores)
            
            # store the accuracy for each epoch
            mean_perf_train = np.mean(pred_train_scores)
            
            pred_scores_history_train.append(mean_perf_train)
            
            print('After Epoch ' + str(i) + ' validation acc / training acc: ' + 
                  str(np.mean(pred_scores)) + ' / ' + str(np.mean(pred_train_scores)))
        
        return pred_scores_history, pred_scores_history_train, loss_history, best_perf, best_ep
        
