# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:51:22 2018

@author: SCUTCYF

Generate samples from a Bernoulli mixture distribution.
Input:
  d: dimension of data
  k: number of components
  n: number of data
Output:
  X: d x n data matrix
  z: 1 x n response variable
  mu: d x k parameters of each Bernoulli component

"""

import numpy as np
from discreteRnd import discreteRnd
def mixBernRnd(d, k, n):
    w = np.ones((1,k))/k
    z = discreteRnd(w,n)
    mu = np.random.rand(d,k)
    X = np.zeros((d,n))
    for i in range(1,k+1):
        idx = np.nonzero(z==i)
        idx = idx[1] # ind must be a tuple refer index
        X[:,idx] = np.random.rand(d,np.sum(z==i))>= np.transpose(np.matlib.repmat(mu[:,i-1],np.sum(z==i),1)) 
        
    return X,z,mu