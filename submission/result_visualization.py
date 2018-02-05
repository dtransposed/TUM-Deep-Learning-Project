import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

###Import data with the training results###
#First architecture
peternagy=pd.read_pickle('trainset20000_100.p')
#Second architecture
damian=pd.read_pickle('Bogunowicz Final big VGG_ 0.001.p')
#Third architecture
spatial_perf_on_val=[0.737,0.827,0.852,0.861,0.872,0.872,0.897,0.855,0.887 ]
spatial_perf_on_train=[0.79175,0.88575,0.93975,0.9575,0.9735,0.97875,0.98,0.97,0.99]
temporal_perf_on_val=[0.35,0.4,0.474,0.519,0.537,0.563,0.555,0.593,0.566]
temporal_perf_on_train=[0.429,0.5655,0.671,0.737,0.80,0.856,0.87,0.89,0.93]

###Create x-axis for the diagrammes
ticks_peter=[]
ticks_damian=[]
ticks_janis=[]

for ticks in range(100):
    ticks_peter.append(ticks)
for ticks in range(1200):
    ticks_damian.append(ticks)
for ticks in range(9):
    ticks_janis.append(ticks)



fig, ax = plt.subplots()
ax.plot(ticks_peter, peternagy['val_acc'] ,color='b')
ax.set(xlabel='Number of epochs',  ylabel=' Validation Accuracy')
ax.grid(color='w', linestyle='-', linewidth=2)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15)
ax.set_title('Single Stream Image Classification',fontsize= 20)
plt.savefig('demo1.png', transparent=True)


fig, ax = plt.subplots()
ax.plot(ticks_damian, damian['pred_hist_train'] ,color='b')
ax.set(xlabel='Number of epochs',  ylabel=' Validation Accuracy')
ax.grid(color='w', linestyle='-', linewidth=2)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15)
ax.set_title('Single Stream Video Classification',fontsize= 20)
plt.savefig('demo2.png', transparent=True)

fig, ax = plt.subplots()
ax.plot(ticks_janis, spatial_perf_on_val,label='Spatial performance on validation set')
ax.plot(ticks_janis, spatial_perf_on_train,label='Spatial performance on training set')
ax.plot(ticks_janis, temporal_perf_on_val,label='Temporal performance on validation set')
ax.plot(ticks_janis, temporal_perf_on_train,label='Temporal performance on training set')
ax.set(xlabel='Number of epochs', ylabel=' Validation Accuracy')
ax.grid(color='w', linestyle='-', linewidth=2)
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15)
ax.legend()
ax.set_ylim([0,1])
ax.set_title('Two-Stream Video Classification',fontsize= 20)
plt.savefig('demo3.png', transparent=True)
