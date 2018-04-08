# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:26:04 2018

@author: SCUTCYF

Variational Bayesian for Gaussian Mixture Model

"""

from mixGaussRnd import mixGaussRnd
from plotClass import plotClass
import numpy as np
from mixGaussVb import mixGaussVb

d = 2
k = 3
n = 2000
X,z = mixGaussRnd(d,k,n)
plotClass(X,z)
m = int(np.floor(n/2))
X1 = X[:,0:m]
X2 = X[:,m-1:-1]
# VB fitting
