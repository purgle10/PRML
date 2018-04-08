# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:04:59 2018

@author: SCUTCYF

Fit linear regression model y=w'x+w0  
Input:
  X: d x n data
  t: 1 x n response
  lambda: regularization parameter
Output:
  model: trained model structure
"""
import numpy as np
import scipy
import sys
sys.path.append("../../common")
from mldivide import mldivide

class model(object):
    __slots__ = ["w", "w0", "xbar", "beta", "U"]
    def __init__(self,w,w0,xbar,beta,U):
        self.w = w
        self.w0 = w0
        self.xbar = xbar
        self.beta = beta
        self.U = U
        
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    if ind.any() < 0:
        ind = -1
    elif (ind.any() >= array_shape[0]*array_shape[1]):
        ind = -1
    return ind

def linReg(X, t, Lambda = 0):
    d = np.size(X, 0)
    idx = np.transpose(np.array(range(0,d)))
    if len(idx) == 0:
        idx = 0
    dg = sub2ind((d,d),idx, idx)
    
    xbar = np.mean(X,1)
    tbar = np.mean(t,1)
    
    X = X - xbar
    t = t - tbar
    
    XX = np.matrix(X)*np.matrix(np.transpose(X))
    XX[dg] = XX[dg] + Lambda
    U = scipy.linalg.cholesky(XX)
    
    Xt = np.matrix(X)*np.matrix(np.transpose(t))
    Unum = mldivide(np.transpose(U), Xt) 
    w = mldivide(U, Unum)
    w0 = tbar - np.dot(w,xbar)
    
    beta = 1/np.mean(np.power(t-np.matrix(np.transpose(w))*np.matrix(X),2))
    return model(w, w0, xbar, beta, U)
