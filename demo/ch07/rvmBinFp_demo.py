# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:48:53 2018

@author: SCUTCYF

RVM for classification

"""

from kmeansRnd import kmeansRnd
from rvmBinFp import rvmBinFp
from rvmBinPred import rvmBinPred
from binPlot import binPlot
import matplotlib.pyplot as plt

k = 2
d = 2
n = 1000

X,t,mu = kmeansRnd(d,k,n)
model,llh = rvmBinFp(X,t-1)
plt.plot(llh)
y,p = rvmBinPred(model,X)
fig = plt.figure()
binPlot(model,X,y)