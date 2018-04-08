# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:37:55 2018

@author: SCUTCYF
"""

import numpy as np
from linRegVb import linRegVb
import matplotlib.pyplot as plt
from linRegPred import linRegPred
from plotCurveBar import plotCurveBar

d = 100
beta = 1e-1
X = np.random.rand(1,d)
w = np.random.randn()
b = np.random.randn()

t = w*X+b+beta*np.random.randn(1,d)
x = np.linspace(np.min(X),np.max(X),d) # test data
model,llh = linRegVb(X,t)
plt.plot(llh)
y,sigma = linRegPred(model,x,t)
plt.figure()
plotCurveBar(x,y,sigma)
plt.hold(True)
plt.plot(X,t,'o')
plt.hold(False)