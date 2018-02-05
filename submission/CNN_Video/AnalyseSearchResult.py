import pickle
from matplotlib import pyplot as plt

results = pickle.load(open('results_300.p','rb'))

for result in results:
    print('Best result on val: ' + str(result['best_perf']) + ' || in epoch: ' + str(result['best_ep'])
          + ' || lr: '+ str(result['lr']) + ' || hidden dim: ' + str(result['hd']))
 
plt.figure()    
for result in results:
    plt.plot(result['pred_hist'])
plt.show()