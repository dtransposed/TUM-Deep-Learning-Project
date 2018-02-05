import torch
import numpy as np
import VideoDataLoader as VDL
import PretrainedCNN
import PredictionHead as PH
import VideoSolver as VS
import VideoClassifier as VC
import glob
from matplotlib import pyplot as plt
import pickle

# Defining hyperparameters
epochs = 50
learning_rate = 0.0001
#learning_rate_CNN = 0.000001
output_classes = 3

####################################################################################################
# train_files = np.array(glob.glob('huge_dataset_trunc*p'))
# val_files = []
# 
# K = train_files.size
# 
# for i in range(0,int(K/4)):
#     index = np.random.randint(0, train_files.size)
#     val_files.append(train_files[index])
#     train_files = np.delete(train_files,index)
#     
# val_files = np.array(val_files)
# 
# train_loader = VDL.iterate_videos_from_pickle(train_files, normalize=True, use_first_hundred=True)
# val_loader = VDL.iterate_videos_from_pickle(val_files, normalize=True, use_first_hundred=True)
####################################################################################################

#path = '/home/peternagy96/Downloads/Frames_for _Damian' 
path = '/home/peternagy96/Project/big_dataset'

#data = VDL.load_videos(path, resize_images=False, huge_data=False)
data = VDL.load_videos(path, resize_images=False, huge_data=False, vid_cap=10)

N = data['targets'].shape[0]

data_train, data_val = VDL.split_dataset(data, size=0.2)

train_loader = VDL.iterate_videos(data_train)
val_loader = VDL.iterate_videos(data_val)

#50176
# define the network that is being used for prediction
pred_head = PH.ThreeLayerFCN([25088,64,output_classes])
#pred_head = PH.ThreeLayerFCN([512,256,output_classes])

#Get fully convolutional network for feature extraction on frame level
pretrained_model = PretrainedCNN.Fully_Conv_Block('vgg11')

for param in pretrained_model.parameters():
    param.requires_grad = False

video_classifier = VC.VideoClassifier(pretrained_model, 'average', pred_head)

optimizer_Classifier = torch.optim.SGD(pred_head.parameters(), lr=learning_rate, momentum=0.9)

solver = VS.VideoSolver(optim_Classifier=optimizer_Classifier)

pred_hist, loss_hist, best_perf, best_ep = solver.train(video_classifier, train_loader, val_loader, N, num_epochs=epochs)

results = {
            'lr': learning_rate,
            'ne': epochs,
            'hd': 64,
            'pred_hist': pred_hist,
            'loss_hist': loss_hist,
            'best_perf': best_perf,
            'best_ep': best_ep
        }

'''
learning_rates = [0.0003,0.0001,0.00003,0.00001]
numbers_epochs = [10]
hidden_dims =  [64,256,2048,4096]

results = []



for lr in learning_rates:
    for ne in numbers_epochs:
        for hd in hidden_dims:
            # define the network that is being used for prediction
            pred_head = PH.ThreeLayerFCN([25088,hd,output_classes])
            #pred_head = PH.ThreeLayerFCN([512,256,output_classes])
            
            #Get fully convolutional network for feature extraction on frame level
            pretrained_model = PretrainedCNN.Fully_Conv_Block('vgg11')
            
            for param in pretrained_model.parameters():
                param.requires_grad = False
            
            video_classifier = VC.VideoClassifier(pretrained_model, 'average', pred_head)
            
            optimizer_Classifier = torch.optim.SGD(pred_head.parameters(), lr=lr, momentum=0.9)
            #optimizer_CNN = torch.optim.SGD(pretrained_model.parameters(), lr=learning_rate_CNN, momentum=0.9)
            
            #solver = VS.VideoSolver(optim_CNN=optimizer_CNN, optim_Classifier=optimizer_Classifier)
            solver = VS.VideoSolver(optim_Classifier=optimizer_Classifier)
            
            pred_hist, loss_hist, best_perf, best_ep = solver.train(video_classifier, train_loader, val_loader, N, num_epochs=ne)
        
            res = {
                    'lr': lr,
                    'ne': ne,
                    'hd': hd,
                    'pred_hist': pred_hist,
                    'best_perf': best_perf,
                    'best_ep': best_ep
                }
            results.append(res)'''

pickle.dump(results,open('results_lr0001_hd64.p','wb'))
'''        
# define the network that is being used for prediction
pred_head = PH.ThreeLayerFCN([25088,256,output_classes])
#pred_head = PH.ThreeLayerFCN([512,256,output_classes])

#Get fully convolutional network for feature extraction on frame level
pretrained_model = PretrainedCNN.Fully_Conv_Block('vgg11')

for param in pretrained_model.parameters():
    param.requires_grad = False

video_classifier = VC.VideoClassifier(pretrained_model, 'average', pred_head)

optimizer_Classifier = torch.optim.SGD(pred_head.parameters(), lr=learning_rate_classifier, momentum=0.9)
#optimizer_CNN = torch.optim.SGD(pretrained_model.parameters(), lr=learning_rate_CNN, momentum=0.9)

#solver = VS.VideoSolver(optim_CNN=optimizer_CNN, optim_Classifier=optimizer_Classifier)
solver = VS.VideoSolver(optim_Classifier=optimizer_Classifier)

pred_hist, loss_hist = solver.train(video_classifier, train_loader, val_loader, N, num_epochs=epochs)

pickle.dump(pred_hist,open('pred_hist.p','wb'))
pickle.dump(loss_hist,open('loss_hist.p','wb'))

plt.figure()
plt.plot(loss_hist)
plt.show()

plt.figure()
plt.plot(pred_hist)
plt.show()'''
