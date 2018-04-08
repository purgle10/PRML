# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:17:24 2018

@author: SCUTCYF
"""

from kmeansRnd import kmeansRnd
from plotClass import plotClass
from kmeans import kmeans
import matplotlib.pyplot as plt

d = 2
k = 4
n = 5000
X,label,mu = kmeansRnd(d,k,n)
y,tmp1, tmp2 = kmeans(X,k)
plotClass(X, label)

plt.figure()
plotClass(X, y)