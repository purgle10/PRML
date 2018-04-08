# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:54:53 2018

@author: SCUTCYF

Generate data from a linear model p(t|w,x)=G(w'x+w0,sigma), sigma=sqrt(1/beta) 
where w and w0 are generated from Gauss(0,1), beta is generated from
Gamma(1,1), X is generated form [0,1].
Input:
  d: dimension of data
  n: number of data
Output:
  X: d x n data matrix
  t: 1 x n response variable
"""
import numpy as np
def linRnd(d,n):
    beta = np.random.gamma(shape=1)
    X = np.random.rand(d,n)
    w = np.random.randn(d,1)
    w0 = np.random.randn(1,)
    t = np.transpose(w).dot(X)+w0+np.random.randn(1,n)/np.sqrt(beta)
    return X, t