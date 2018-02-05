#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:26:22 2018

@author: peternagy96
"""

import pickle
from matplotlib import pyplot as plt
import seaborn as sb

results = pickle.load(open('final_results_0.332.p','rb'))

plt.figure()

sb.set_style('darkgrid')
pred_hist = sb.load_dataset(results['pred_hist'])
sb.plot(pred_hist)

plt.show()