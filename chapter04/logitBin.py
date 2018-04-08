# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:28:53 2018

@author: SCUTCYF

Logistic regression for binary classification optimized by Newton-Raphson method.
Input:
  X: d x n data matrix
  z: 1 x n label (0/1)
  lambda: regularization parameter
  eta: step size
Output:
  model: trained model structure
  llh: loglikelihood


"""
import numpy as np
import sys
sys.path.append("../../common")
from mldivide import mldivide
from log1pexp import log1pexp

class model(object):
    __slots__ = ["w"]
    def __init__(self,w):
        self.w = w

def logitBin(X, y, Lambda=1e-4, eta=1e-1):
    (d,n) = np.shape(X)
    X = np.concatenate((X,np.ones((1,n))),0)
    (d,n) = np.shape(X)
    tol = 1e-4
    epoch = 200
    inf = 1000000
    llh = np.ones(epoch)*(-inf)
    h = 2*y-1
    w = np.random.rand(d,1)
    for t in range(1,epoch):
        a = np.transpose(w).dot(X)
        llh[t] = -(np.sum(log1pexp(-h*a))+0.5*Lambda*np.multiply(w,w).sum(axis=0))/n
        if(llh[t]-llh[t-1] < tol):
            break
        z = np.exp(-log1pexp(-a)) # z = sigmoid(a); in matlab
        g = X.dot(np.transpose(z-y))+Lambda*w
        r = z*(1-z)
        Xw = X*np.sqrt(r)
        H = Xw.dot(Xw.transpose())+Lambda*np.eye(d)
        w = w-eta*mldivide(H,g)
        
    llh = llh[1:t]
    model.w = w
    return model,llh