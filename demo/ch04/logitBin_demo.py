# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:21:58 2018

@author: SCUTCYF
demos for ch04
"""

# Logistic logistic regression for binary classification
import sys
sys.path.append("../../chapter04")
from kmeansRnd import kmeansRnd
from logitBin import logitBin
import matplotlib.pyplot as plt
from logitBinPred import logitBinPred
from binPlot import binPlot

d = 2
k = 2

n = 1000
X,y,mu = kmeansRnd(d,k,n)
model,llh = logitBin(X,y-1)
plt.plot(llh)
y,p = logitBinPred(model,X)
t =y +1
fig = plt.figure()
binPlot(model,X,y) # plt.scatter is slow version, should change to faster implementation
# and the function discreteRnd() has a better way to implement