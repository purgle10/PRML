# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:42:57 2018

@author: SCUTCYF

regression

"""
import numpy as np
from rvmRegFp import rvmRegFp
from linRegPred import linRegPred
import sys
sys.path.append("..\..\common")
from plotCurveBar import plotCurveBar
import matplotlib.pyplot as plt

d = 100
beta = 1e-1
X = np.random.rand(1,d)
w = np.random.randn()
b = np.random.randn()
t = w*X+beta*np.random.randn(1,d)
x = np.linspace(np.min(X),np.max(X),num=d) # test data


# RVM regression by Mackay fix point update
model,llh = rvmRegFp(X,t)
plt.plot(llh)
y,sigma = linRegPred(model,x,t)
plt.figure()
plotCurveBar(x.reshape(y.shape),y,sigma.reshape(y.shape))
mycolor = [255/255,228/255,225/255] # pink
plt.hold(True)
plt.plot(X,t,'o')
plt.hold(False)