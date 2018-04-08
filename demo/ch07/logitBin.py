# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:38:21 2018

@author: SCUTCYF

this function is not totally same as logitBin() in chapter04

"""

import numpy as np
import scipy
import sys
sys.path.append("../../common")
from mldivide import mldivide
from log1pexp import log1pexp

     
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    if ind.any() < 0:
        ind = -1
    elif (ind.any() >= array_shape[0]*array_shape[1]):
        ind = -1
    return ind

def logitBin(X, t, Lambda, w):
    # Logistic regression
    (d,n) = np.shape(X)
    tol = 1e-4
    maxiter = 100
    inf = 1000000
    llh = np.ones(maxiter)*(-inf)
    idx = np.array(range(0,d))
    if len(idx) == 0:
        idx = 0
    dg = sub2ind((d,d),idx,idx)
    h = np.ones((1,n))
    h[t==0] = -1
    a = np.transpose(w).dot(X)
    for iter in range(1,maxiter):
        y = np.exp(-log1pexp(-a)) # y = sigmoid(a);
        r = y*(1-y)
        Xw = X*np.sqrt(r)
        H = Xw.dot(Xw.transpose())
        H.flatten()[dg] = H.flatten()[dg] +Lambda
        U = scipy.linalg.cholesky(H)
        g = X.dot((y-t).transpose()).reshape((d,)) + Lambda*w
        p = -mldivide(U,mldivide(U.transpose(),g))
        wo = w
        w = wo+p
        a = w.transpose().dot(X)
        
        llh[iter] = -np.sum(log1pexp(-h*a))-0.5*(Lambda*np.multiply(w,w)).sum()
        incr = llh[iter]-llh[iter-1]
        while(incr < 0):
            p = p/2
            w = wo+p
            a = w.transpose().dot(X)
            llh[iter] = -np.sum(log1pexp(-h*a))-0.5*(Lambda*np.multiply(w,w)).sum()
            incr = llh[iter]-llh[iter-1]
        if incr < tol:
           break
        
    llh = llh[1:iter]
    return w,llh,U