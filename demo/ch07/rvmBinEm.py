# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:29:18 2018

@author: SCUTCYF

Relevance Vector Machine (ARD sparse prior) for binary classification.
trained by empirical bayesian (type II ML) using EM.
Input:
  X: d x n data matrix
  t: 1 x n label (0/1)
  alpha: prior parameter
Output:
  model: trained model structure
  llh: loglikelihood

"""
import numpy as np
from logitBin import logitBin

class model(object):
    __slots__ = ["index","w","alpha"]
    def __init__(self,index,w,alpha):
        self.index = index
        self.w = w
        self.alpha = alpha

def rvmBinEm(X, t, alpha = 1):
    n = X.shape[1]
    X = np.concatenate((X,np.ones((1,n))),0)
    d = X.shape[0]
    alpha = alpha*np.ones((d,))
    m = np.zeros((d,))
    
    tol = 1e-4
    maxiter = 100
    inf = 1000000
    llh = -inf*np.ones(maxiter)
    index = np.array(range(0,d))
    for iter in range(1,maxiter):
        # remove zeros
        nz = 1/alpha > tol
        nz = np.nonzero(nz)
        index = index[nz]
        alpha = alpha[nz]
        X = X[nz]
        m = m[nz]
        m,e,U = logitBin(X,t,alpha,m)
        
        m2 = m*m
        llh[iter] = e[-1]+0.5*(np.sum(np.log(alpha))-2*np.sum(np.log(np.diag(U)))-np.multiply(alpha,m2).sum()-n*np.log(2*np.pi))
        if (np.abs(llh[iter]-llh[iter-1])<tol*np.abs(llh[iter-1])):
            break
        
        V = np.linalg.inv(U)
        dgS = np.multiply(V,V).sum(axis=1)
        alpha = 1/(m2+dgS)
        
    llh = llh[1:iter]
    
    model.index = index
    model.w = m
    model.alpha = alpha
    return model,llh