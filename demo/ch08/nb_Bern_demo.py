# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:50:20 2018

@author: SCUTCYF

Naive Bayes with independent Bernoulli

"""
import numpy as np
from mixBernRnd import mixBernRnd
from nbBern import nbBern
from nbBernPred import nbBernPred
d = 10
k = 2
n = 2000
X,t,mu = mixBernRnd(d,k,n)
m = int(np.floor(n/2))
X1 = X[:,0:m]
X2 = X[:,m-1:-1]
t1 = t[:,0:m]
t2 = t[:,m-1:-1]
model = nbBern(X1, t1)
y2 = nbBernPred(model, X2)
err = np.sum(t2!=(y2+1))/t2.size