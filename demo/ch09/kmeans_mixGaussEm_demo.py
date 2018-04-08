# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:17:52 2018

@author: SCUTCYF

Gauss mixture initialized by kmeans

"""

from kmeans import kmeans
from mixGaussRnd import mixGaussRnd
from mixGaussEm import mixGaussEm
from plotClass import plotClass
import matplotlib.pyplot as plt

d = 2
k = 3
n = 500
X,label,tmp = mixGaussRnd(d,k,n)
init,energy,model_means = kmeans(X, k)
z,model,llh = mixGaussEm(X, init)
plotClass(X,label)
plt.figure()
plotClass(X,init)
plt.figure()
plotClass(X,z)
plt.figure()
plt.plot(llh)