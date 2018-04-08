# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:58:09 2018

@author: SCUTCYF

demos for ch03
"""
import sys
sys.path.append("../../chapter01")
from linRnd import linRnd 
from linRegEm import linRegEm
from linRegPred import linRegPred
from plotCurveBar import plotCurveBar
import matplotlib.pyplot as plt

d = 1
n = 200
x,t = linRnd(d,n)

# Empirical Bayesian linear regression via EM
model,llh = linRegEm(x,t)
plt.plot(llh)
y,sigma = linRegPred(model,x,t)
plt.Figure
plotCurveBar(x,y,sigma)

mycolor = [255/255,228/255,225/255] # pink
plt.hold(True)
plt.plot(x,t,'o',markerfacecolor='none',markeredgecolor=mycolor)
plt.hold(False)