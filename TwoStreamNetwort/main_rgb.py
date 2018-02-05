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


vid_cap = 1000
path = '/home/peternagy96/Project/big_dataset/'
N = 5*vid_cap

train_paths,val_paths = VDL.split_path(path,vid_cap,0.1)

pickle.dump([train_paths,val_paths],open('train_test_paths.p','wb'))

print('Training set consists of ' + str(len(train_paths)) + ' videos.')
print('Validation set consists of ' + str(len(val_paths)) + ' videos.')

train_loader = VDL.load_videos_path(train_paths,batchsize=1)
val_loader = VDL.load_videos_path(val_paths,batchsize=1)

# =============================================================================
# class MeanModule(nn.Module):
#     def __init__(self):
#         super(MeanModule, self).__init__()
#     
#     def forward(self, x):
#         outputs = x.mean(0)
#         if len(outputs.cpu().data.numpy().shape) == 1:
#             outputs = outputs.view(1,outputs.shape[0])
#         return outputs
# =============================================================================

#Use smaller network -> CNNFLOW
model_rgb = CNNflow.flowCNN(input_channel=3)
num_ftrs = model_rgb.classifier.in_features
model_rgb.classifier = nn.Linear(num_ftrs, 5)

#CNN OPTFLOW
model_flow = CNNflow.flowCNN()
num_ftrs = model_flow.classifier.in_features
model_flow.classifier = nn.Linear(num_ftrs, 5)

models = [model_rgb,model_flow]

if use_gpu:
    model_rgb = model_rgb.cuda()
    model_flow = model_flow.cuda()
    print("\nModel loaded in CUDA")

# optimizer
optimizer_flow = optim.Adam(model_flow.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optims = [optimizer_rgb,optimizer_flow]

# train model
solver = VS.VideoSolver(optim=optims)
pred_scores_history, pred_scores_history_train, loss_history, best_perf, best_ep = solver.train(models, train_loader, val_loader, N, num_epochs=5, 
                                                                                                train_CNN=False, train_rgb=True, train_opt=False, pure_train=True)
# dump result
pickle.dump([pred_scores_history, pred_scores_history_train, loss_history],open('5samples.p','wb'))


