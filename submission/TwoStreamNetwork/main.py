from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import VideoDataLoader1 as VDL
import VideoSolver as VS
import CNNflow
import PretrainedCNN as PC
import PredictionHead as PH
import VideoClassifier as VC
import pickle

plt.ion()   # interactive mode

use_gpu = torch.cuda.is_available()


vid_cap = 10
path = '/home/peternagy96/Project/big_dataset/'
#path = '/home/peternagy96/Project/dataset_peter/'
N = 5*vid_cap

train_paths,val_paths = VDL.split_path(path,vid_cap,0.2)

print('Training set consists of ' + str(len(train_paths)) + ' videos.')
print('Validation set consists of ' + str(len(val_paths)) + ' videos.')

train_loader = VDL.load_videos_path(train_paths,batchsize=1)
val_loader = VDL.load_videos_path(val_paths,batchsize=1)

class MeanModule(nn.Module):
    def __init__(self):
        super(MeanModule, self).__init__()
    
    def forward(self, x):
        outputs = x.mean(0)
        if len(outputs.cpu().data.numpy().shape) == 1:
            outputs = outputs.view(1,outputs.shape[0])
        return outputs

#CNN RGB
# =============================================================================
# model_rgb = models.densenet121(pretrained=True)
# for param in model_rgb.features.parameters():
#             param.requires_grad = False
# num_ftrs = model_rgb.classifier.in_features
# model_rgb.classifier = nn.Sequential(MeanModule(),
#                                      nn.Linear(num_ftrs, 5),
#                                      nn.Softmax(dim=1))
# =============================================================================

#Use smaller network -> CNNFLOW
model_rgb = CNNflow.flowCNN(input_channel=3)
num_ftrs = model_rgb.classifier.in_features
model_rgb.classifier = nn.Linear(num_ftrs, 5)

#CNN OPTFLOW
model_flow = CNNflow.flowCNN()
num_ftrs = model_flow.classifier.in_features
model_flow.classifier = nn.Linear(num_ftrs, 5)

# =============================================================================
# cnn = PC.Fully_Conv_Block('densenet121')
# fcn = PH.ThreeLayerFCN([50176,512,5])
# vid_model = VC.VideoClassifier(cnn, 'average', fcn)
# 
# for param in vid_model.CNN_Model.parameters():
#     param.requires_grad = False
# =============================================================================
    
models = [model_rgb,model_flow]

#LOAD MODEL PARAMETERS IF NEEDED
#model_ft.load_state_dict(torch.load("models/test.pth.tar"))

if use_gpu:
    model_rgb = model_rgb.cuda()
    model_flow = model_flow.cuda()
    print("\nModel loaded in CUDA")

#optimizer_rgb = torch.optim.SGD(vid_model.Classifier.parameters(), lr=0.001,  momentum=0.9)
#optimizer_rgb = torch.optim.SGD(vid_model.Classifier.parameters(), lr=0.001,  momentum=0.9)
optimizer_flow = optim.Adam(model_flow.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#solver = train.Solver(optimizer_rgb=optimizer_rgb, optimizer_flow=optimizer_flow)
#model_rgb, model_flow = solver.train_model(model_flow, model_rgb, trainl_rgb, vall_rgb, trainl_flow, vall_flow, N_train, N_val, num_epochs=25)
optims = [optimizer_rgb,optimizer_flow]

solver = VS.VideoSolver(optim=optims)
pred_scores_history, pred_scores_history_train, loss_history, best_perf, best_ep = solver.train(models, train_loader, val_loader, N, num_epochs=20, 
                                                                                                train_CNN=False, train_rgb=False, train_opt=True, pure_train=False)

pickle.dump([pred_scores_history, pred_scores_history_train, loss_history],open('5samples.p','wb'))


