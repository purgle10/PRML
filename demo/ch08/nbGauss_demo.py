# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:22:10 2018

@author: SCUTCYF

Naive Bayes with Gaussian

"""

import numpy as np
from plotClass import plotClass
from kmeansRnd import kmeansRnd
from nbGauss import nbGauss
from nbGaussPred import nbGaussPred
import matplotlib.pyplot as plt
d = 2
k = 3
n = 1000
X,t,mu = kmeansRnd(d,k,n)
plotClass(X, t)

m = int(np.floor(n/2))
X1 = X[:,0:m]
X2 = X[:,m-1:-1]
t1 = t[:,0:m]

model = nbGauss(X1,t1)
y2 = nbGaussPred(model,X2)

plt.figure()
plotClass(X2, y2+1)