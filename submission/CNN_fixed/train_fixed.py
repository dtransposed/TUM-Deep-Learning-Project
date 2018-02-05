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

class Solver():
    
    default_sgd_args = {"lr": 0.001,
                     "momentum": 0.9,
                     "weight_decay": 0.0}
    
    def __init__(self, optimizer=optim.SGD, optim_args={}, loss_func=torch.nn.CrossEntropyLoss()):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        self._reset_histories()

    def _reset_histories(self):
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        

    def train_model(self, model, train_loader, val_loader, N_train, N_val, num_epochs):
        print("\n# of training samples: ", N_train)
        print("# of validation samples: ", N_val, "\n")
        self._reset_histories()
        since = time.time()
    
        best_model_wts = model.state_dict()
        best_acc = 0.0
    
        #EPOCHs
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            
            
            #TRAINING
            epoch_time = time.time()
            
            #self.scheduler.step()
            model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in train_loader:
                
                inputs, targets = data
                
                if torch.cuda.is_available():
                    #print("CUDA is available")
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward              
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                self.optimizer.step()
                

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / N_train
            epoch_acc = running_corrects / N_train
            train_acc = epoch_acc
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss, epoch_acc))
            train_time = time.time() - epoch_time
            print('Validation took {:.0f}m {:.0f}s'.format(
            train_time // 60, train_time % 60))

                
            #VALIDATION
            after_train = time.time()
            model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in val_loader:
                
                inputs, targets = data
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = self.loss_func(outputs, targets)

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / N_val
            epoch_acc = running_corrects / N_val
            self.val_acc_history.append(epoch_acc)
            self.val_loss_history.append(epoch_loss)
            if train_acc > 0.8 and epoch_acc < 0.6:
                print("Training stopped at epoch "+str(epoch+1))
                return model
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "val", epoch_loss, epoch_acc))
            val_time = time.time() - after_train
            print('Validation took {:.0f}m {:.0f}s'.format(
            val_time // 60, val_time % 60))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    def test_model(self, model, test_loader, N_test):
        print("\n# of test samples: ", N_test, "\n")
        self._reset_histories()
        
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in test_loader:
            
            inputs, targets = data
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.loss_func(outputs, targets)

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / N_test
            epoch_acc = running_corrects / N_test
            self.val_acc_history.append(epoch_acc)
            self.val_loss_history.append(epoch_loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "val", epoch_loss, epoch_acc))
    
            print()

        print('Test Accuracy: {:4f}'.format(epoch_acc))
        return model
    