# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:23:35 2018

@author: SCUTCYF

Variational Bayesian inference for Gaussian mixture.
Input: 
  X: d x n data matrix
  m: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or model structure
Output:
  label: 1 x n cluster label
  model: trained model structure
  L: variational lower bound
Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)

"""

import numpy as np
import scipy

class Prior(object):
    __slots__ = ["alpha","kappa","m","v","M","logW"]
    def __init__(self,alpha,kappa,m,v,M,logW):
        self.alpha = alpha
        self.kappa = kappa
        self.m = m
        self.v = v
        self.M = M
        self.logW = logW

def mixGaussVb(X, m, Prior=0):
    print("Variational Bayesian Gaussian mixture: running ... \n")
    (d,n)= X.shape
    if Prior == 0:
        Prior.alpha = 1
        Prior.kappa = 1
        Prior.m = np.mean(X,1)
        Prior.v = d+1
        Prior.M = np.eye(d)   # M = inv(W)

    Prior.logW = -2*np.sum(np.log(np.diag(scipy.linalg.cholesky(Prior.M))))
    
    tol = 1e-8
    maxiter = 2000
    inf = 10000000
    L = -inf*np.ones((maxiter+1,1))
    # model = 
        