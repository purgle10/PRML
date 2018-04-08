# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:57:21 2018

@author: SCUTCYF

regression

"""

import numpy as np
import sys
sys.path.append("..\..\common")
from plotCurveBar import plotCurveBar
from rvmRegEm import rvmRegEm
from linRegPred import linRegPred
import matplotlib.pyplot as plt

d = 100
beta = 1e-1
X = np.random.rand(1,d)
w = np.random.randn(1)
b = np.random.randn(1)
t = w*X+beta*(np.random.randn(1,d))
x = np.linspace(np.min(X),np.max(X),d) # test data

# RVM regression by EM
model,llh = rvmRegEm(X,t)
plt.plot(llh)
plt.figure()
y,sigma = linRegPred(model,x,t)
plotCurveBar(x.reshape(y.shape),y,sigma.reshape(y.shape))
mycolor = [255/255,228/255,225/255] # pink
plt.hold(True)
plt.plot(X,t,'o')
plt.hold(False)