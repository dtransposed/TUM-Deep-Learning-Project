#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch.optim as optim
import torch.nn
import torch.cuda
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import VideoDataLoader1 as VDL

class Solver():
    
    default_sgd_args = {"lr": 0.001,
                     "momentum": 0.9,
                     "weight_decay": 0.0}
    
    def __init__(self, optimizer_rgb=optim.Adam, optimizer_flow=optim.Adam, optim_args={}, loss_func=torch.nn.CrossEntropyLoss()):
        self.optimizer_rgb = optimizer_rgb   
        self.scheduler_rgb = lr_scheduler.StepLR(optimizer_rgb, step_size=7, gamma=0.1)
        
        self.optimizer_flow = optimizer_flow
        self.scheduler_flow = lr_scheduler.StepLR(optimizer_flow, step_size=7, gamma=0.1)
        
        self.loss_func = loss_func
        
        self._reset_histories()

    def _reset_histories(self):
        #RGB
        self.train_loss_history_rgb = []
        self.train_acc_history_rgb = []
        self.val_acc_history_rgb = []
        self.val_loss_history_rgb = []
        #FLOW
        self.train_loss_history_flow = []
        self.train_acc_history_flow = []
        self.val_acc_history_flow = []
        self.val_loss_history_flow = []
        #OVERALL
# =============================================================================
#         self.train_loss_history = []
#         self.train_acc_history = []
#         self.val_acc_history = []
#         self.val_loss_history = []
# =============================================================================
        
#    def train_stop(self):
#        return None

    def train_model(self,  model_flow, model_rgb, train_loader, val_loader,train_flow , val_flow, N_train, N_val, num_epochs):
        #dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
        #dataset_size_train = train_loader['targets'].shape[0]
        #dataset_size_val = val_loader['targets'].shape[0]
        print("\n# of training samples: ", N_train)
        print("# of validation samples: ", N_val, "\n")
        self._reset_histories()
        since = time.time()
    
        best_model_rgb = model_rgb.state_dict()
        best_model_flow = model_flow.state_dict()
        best_acc_rgb = 0.0
        best_acc_flow = 0.0
    
        #EPOCHs
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)    
            epoch_time = time.time()
            
            #TRAINING----------
            self.scheduler_rgb.step()
            self.scheduler_flow.step()
            model_rgb.train(True)  # Set model to training mode
            model_flow.train(True)  # Set model to training mode

            running_loss_rgb = 0.0
            running_corrects_rgb = 0
            running_loss_flow = 0.0
            running_corrects_flow = 0
            running_corrects = 0

            #RGB TRAINING
            #  for data in train_loader:
            self.optimizer_rgb.zero_grad()
            
            for j, (inputs, targets) in enumerate(train_loader.__next__()):
                
                #inputs, targets = data
                #print(targets.shape)
                #inputs = np.swapaxes(inputs,1,2)
                # wrap them in Variable
                if torch.cuda.is_available():
                    #print("CUDA is available")
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)
                
                # zero the parameter gradients
                #self.optimizer_rgb.zero_grad()

                # forward
                outputs = model_rgb(inputs)
                #outputs = torch.nn.Softmax()
                #guess_rgb = torch.mean(outputs,dim=0)
                
                outputs = outputs.mean(0)
                
                if len(outputs.cpu().data.numpy().shape) == 1:
                    outputs = outputs.view(1,outputs.shape[0])
                
                guess_rgb = torch.max(outputs.data)
                
                
                _, preds_rgb = torch.max(outputs.data, 1)
                
                
                #print(outputs)
                #print(targets.data)
                
                loss_rgb = self.loss_func(outputs, targets)
                
                #print(loss_rgb)
                
                #print("OUTPUTS####")
                #print(outputs)
                #print("PREDS####")
                #print(preds)
                # backward + optimize only if in training phase
                loss_rgb.backward()
                
                #self.optimizer_rgb.step()
                
                #for i,param in enumerate(model_rgb.parameters()):
                    #if i == 1:
                        #print(param[0].mean())

                # statistics
                running_loss_rgb += loss_rgb.data[0]
                running_corrects_rgb += torch.sum(preds_rgb == targets.data)
            self.optimizer_rgb.step()
                
                #FLOW TRAINING
            '''for j, (optflow, targets) in enumerate(train_flow.__next__()):
                # zero the parameter gradients
                if torch.cuda.is_available():
                    #print("CUDA is available")
                    optflow = Variable(optflow.cuda()).type(torch.cuda.FloatTensor)
                    targets = Variable(targets.cuda())
                else:
                    optflow, targets = Variable(optflow), Variable(targets)
                self.optimizer_flow.zero_grad()

                # forward
                optflow = optflow.view(7,10,224,224)
                outputs = model_flow(optflow)
                outputs = outputs.mean(0)
                if len(outputs.cpu().data.numpy().shape) == 1:
                    outputs = outputs.view(1,outputs.shape[0])
                guess_flow = torch.max(outputs.data)
                _, preds_flow = torch.max(outputs.data, 1)
                loss_flow = self.loss_func(outputs, targets)
                #print("OUTPUTS####")
                #print(outputs)
                #print("PREDS####")
                #print(preds)
                # backward + optimize only if in training phase
                loss_flow.backward()
                self.optimizer_flow.step()
                

                # statistics
                running_loss_flow += loss_flow.data[0]
                
                running_corrects_flow += torch.sum(preds_flow == targets.data)
                
                if guess_flow >= guess_rgb:
                    running_corrects += torch.sum(preds_flow == targets.data)
                else:
                    running_corrects += torch.sum(preds_rgb == targets.data)'''

            #STATISTICS----------
            epoch_loss_rgb = running_loss_rgb / N_train
            epoch_acc_rgb = running_corrects_rgb / N_train
            self.train_loss_history_rgb.append(epoch_loss_rgb)
            self.train_acc_history_rgb.append(epoch_acc_rgb)
            
            epoch_loss_flow = running_loss_flow / N_train
            epoch_acc_flow = running_corrects_flow / N_train
            self.train_loss_history_flow.append(epoch_loss_flow)
            self.train_acc_history_flow.append(epoch_acc_flow)

            print('{} RGB Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss_rgb, epoch_acc_rgb))
            print('{} FLOW Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss_flow, epoch_acc_flow))
            train_time = time.time() - epoch_time
            print('Training took {:.0f}m {:.0f}s'.format(
            train_time // 60, train_time % 60))

                
            #VALIDATION----------
            after_train = time.time()
            model_rgb.train(False)  # Set model to evaluate mode
            model_flow.train(False) 

            running_loss_rgb = 0.0
            running_corrects_rgb = 0
            running_loss_flow = 0.0
            running_corrects_flow = 0
            running_corrects = 0
            ''' 
            # Iterate over data.
            for j, (inputs, targets) in enumerate(val_loader.__next__()):
                
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)

                # zero the parameter gradients
                #self.optimizer_rgb.zero_grad()

                # forward
                outputs = model_rgb(inputs)
                outputs = outputs.mean(0)
                if len(outputs.cpu().data.numpy().shape) == 1:
                    outputs = outputs.view(1,outputs.shape[0])
                _, preds_rgb = torch.max(outputs.data, 1)
                loss_rgb = self.loss_func(outputs, targets)

                # statistics
                running_loss_rgb += loss_rgb.data[0]
                running_corrects_rgb += torch.sum(preds_rgb == targets.data)
                
            #FLOW VALIDATION
            for j, (inputs, targets) in enumerate(val_flow.__next__()):
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda()).type(torch.cuda.FloatTensor)
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)

                # zero the parameter gradients
                #self.optimizer_flow.zero_grad()

                # forward
                inputs = inputs.view(7,10,224,224)
                outputs = model_flow(inputs)
                outputs = outputs.mean(0)
                if len(outputs.cpu().data.numpy().shape) == 1:
                    outputs = outputs.view(1,outputs.shape[0])
                _, preds_flow = torch.max(outputs.data, 1)
                loss_flow = self.loss_func(outputs, targets)

                # statistics
                running_loss_flow += loss_flow.data[0]
                running_corrects_flow += torch.sum(preds_flow == targets.data)'''

            epoch_loss_rgb = running_loss_rgb / N_val
            epoch_acc_rgb = running_corrects_rgb / N_val
            self.train_loss_history_rgb.append(epoch_loss_rgb)
            self.train_acc_history_rgb.append(epoch_acc_rgb)
            
            epoch_loss_flow = running_loss_flow / N_train
            epoch_acc_flow = running_corrects_flow / N_train
            self.train_loss_history_flow.append(epoch_loss_flow)
            self.train_acc_history_flow.append(epoch_acc_flow)
            
            print('{} RGB Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss_rgb, epoch_acc_rgb))
            print('{} FLOW Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss_flow, epoch_acc_flow))
            
            val_time = time.time() - after_train
            print('Validation took {:.0f}m {:.0f}s'.format(
            val_time // 60, val_time % 60))
            
            # deep copy the model
            if epoch_acc_rgb > best_acc_rgb:
                best_acc_rgb = epoch_acc_rgb
                best_model_rgb = model_rgb.state_dict()
                
            if epoch_acc_flow > best_acc_flow:
                best_acc_flow = epoch_acc_flow
                best_model_flow = model_flow.state_dict()
                
#            if self.train_stop():
#                print("Training finished in "+(epoch+1)+" epochs.")
#                break
    
            print()
    
        time_elapsed = time.time() - since
        if best_acc_rgb >= best_acc_flow:
            best_acc = best_acc_rgb
        else:
            best_acc = best_acc_flow
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model_rgb.load_state_dict(best_model_rgb)
        model_flow.load_state_dict(best_model_flow)
        return model_rgb, model_flow
    