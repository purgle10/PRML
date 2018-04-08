# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:13:22 2018

@author: SCUTCYF
"""


from kmeansRnd import kmeansRnd
from plotClass import plotClass
from kmedoids import kmedoids
import matplotlib.pyplot as plt

d = 2
k = 3
n = 5000
X,label,mu = kmeansRnd(d,k,n)
y,energy, index = kmedoids(X,k)
plotClass(X,label)
plt.figure()
plotClass(X,y)