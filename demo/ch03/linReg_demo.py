# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:50:25 2018

@author: SCUTCYF

demos for ch03
"""
import sys
sys.path.append("../../chapter03")
sys.path.append("../../common")
from linRnd import linRnd
from linReg import linReg
from linRegPred import linRegPred
from plotCurveBar import plotCurveBar
import matplotlib.pyplot as plt

d = 1
n = 200
x,t = linRnd(d,n)

mycolor = [255/255,228/255,225/255] # pink

# Linear regression
Model = linReg(x, t)
y,sigma =linRegPred(Model,x,t,2)
sigma = sigma.reshape(x.shape)
plotCurveBar(x,y,sigma)
plt.hold(True)
plt.plot(x,t,'o',markerfacecolor='none',markeredgecolor=mycolor)
plt.hold(False)