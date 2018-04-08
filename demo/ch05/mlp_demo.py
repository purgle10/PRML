# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:08:18 2018

@author: SCUTCYF

mlp_demo ch05
"""
import sys
sys.path.append("../../chapter05")
import numpy as np
from mlp import mlp
import matplotlib.pyplot as plt
from mlpPred import mlpPred
h = np.array([4,5])
X = np.array([[0,0,1,1],[0,1,0,1]])
T = np.array([[0,1,1,0]])

model,mse = mlp(X,T,h)
plt.plot(mse)
print('T = ',T)
Y = mlpPred(model,X)
print('Y = ',Y)