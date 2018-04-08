# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:25:31 2018

@author: SCUTCYF

RVM for classification

"""

from kmeansRnd import kmeansRnd
from rvmBinEm import rvmBinEm
from rvmBinPred import rvmBinPred
from binPlot import binPlot
import matplotlib.pyplot as plt

k = 2
d = 2
n = 1000

X,t,mu = kmeansRnd(d,k,n)
model,llh = rvmBinEm(X,t-1)
plt.plot(llh)
y,p = rvmBinPred(model,X)
# y = y+1 # don't need +1 in python
fig = plt.figure()
binPlot(model,X,y)