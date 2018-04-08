# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:41:28 2018

@author: SCUTCYF

Variational Bayesian inference for linear regression.
Input:
  X: d x n data
  t: 1 x n response
  prior: prior parameter
Output:
  model: trained model structure
  energy: variational lower bound

"""

import numpy as np
import scipy
from mldivide import mldivide
from scipy.special import gammaln

class Prior(object):
    __slots__ = ["a0","b0","c0","d0"]
    def __init__(self,a0,b0,c0,d0):
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
        
class model(object):
    __slots__ = ["w0","w","alpha","beta","a","b","c","d","xbar"]
    def __init__(self,w0,w,alpha,beta,a,b,c,d,xbar):
        self.w0 = w0
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.xbar = xbar
        
def linRegVb(X, t, Prior=0):
    (m,n)= X.shape
    if Prior==0:
        a0 = 1e-4
        b0 = 1e-4
        c0 = 1e-4
        d0 = 1e-4
    else:
        a0 = Prior.a0
        b0 = Prior.b0
        c0 = Prior.c0
        d0 = Prior.d0
        
    I = np.eye(m)
    xbar = np.mean(X, 1)
    tbar = np.mean(t, 1)
    
    X = X-xbar.reshape(xbar.size,1)
    t = t-tbar.reshape(tbar.size,1)
    
    XX = X.dot(X.transpose())
    Xt = X.dot(t.transpose())
    
    tol = 1e-8
    maxiter = 100
    inf = 10000000
    energy = -inf*np.ones((maxiter+1,1))
    
    a = a0 + m/2
    c = c0 + n/2
    Ealpha = 1e-4
    Ebeta = 1e-4
    
    for iter in range(1,maxiter):
        invS = np.diag([Ealpha])+Ebeta*XX
        U = scipy.linalg.cholesky(invS)
        Ew = Ebeta*(mldivide(U, mldivide(U.transpose(), Xt)))
        KLw = -np.sum(np.log(np.diag(U)))
        
        w2 = np.multiply(Ew,Ew).sum()
        invU = mldivide(U, I)
        trS = np.multiply(invU, invU).sum()
        b = b0+0.5*(w2+trS)
        Ealpha = a/b
        KLalpha = -a*np.log(b)
        e2pre = t-Ew.transpose().dot(X)
        e2 = np.sum(np.multiply(e2pre,e2pre))
        invUX = mldivide(U, X)
        trXSX = np.multiply(invUX, invUX).sum()
        d = d0+0.5*(e2+trXSX)
        Ebeta = c/d
        KLbeta = -c*np.log(d)
        energy[iter] = KLalpha +KLbeta + KLw
        if energy[iter]-energy[iter-1] < tol*np.abs(energy[iter-1]):
            break
        
    const = gammaln(a)-gammaln(a0)+gammaln(c)-gammaln(c0)+a0*np.log(b0)\
            + c0*np.log(d0)+0.5*(m-n*np.log(2*np.pi))
    energy = energy[1:iter]+const
    w0 = tbar-np.multiply(Ew,xbar).sum()
    
    model.w0 = w0
    model.w = Ew
    model.alpha = Ealpha
    model.beta = Ebeta
    model.a = a
    model.b = b
    model.c = c
    model.d = d
    model.xbar = xbar
    return model, energy
    
    