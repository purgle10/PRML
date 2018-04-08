# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:00:21 2018

@author: SCUTCYF

Fit empirical Bayesian linear regression model with EM (p.448 chapter 9.3.4)
Input:
  X: d x n data
  t: 1 x n response
  alpha: prior parameter
  beta: prior parameter
Output:
  model: trained model structure
  llh: loglikelihood
  
"""
import numpy as np
import scipy
import sys
sys.path.append("../../common")
from mldivide import mldivide

class model(object):
    __slots__ = ["w", "w0", "alpha","xbar", "beta", "U"]
    def __init__(self,w,w0,alpha,xbar,beta,U):
        self.w = w
        self.w0 = w0
        self.alpha = alpha
        self.xbar = xbar
        self.beta = beta
        self.U = U

def linRegEm(X,t,alpha=0.02,beta=0.5):
    (d,n)=np.shape(X)
    
    xbar = np.mean(X, 1)
    tbar = np.mean(t, 1)
    
    X = X-xbar.reshape(xbar.size,1)
    t = t-tbar.reshape(tbar.size,1)
    
    XX = X.dot(X.transpose())
    Xt = X.dot(t.transpose())
    
    tol = 1e-4
    maxiter = 100
    inf = -10000000
    llh = -inf*np.ones((maxiter+1,1))
    
    for iter in range(1,maxiter):
        A = beta*XX + alpha*np.eye(d)
        U = scipy.linalg.cholesky(A)
        
        m = beta*(mldivide(U,mldivide(U.transpose(),Xt)))
        m2 = np.multiply(m,m).sum(axis=0) # dot(m,m) in matlab
        e2 = np.power(t-m.transpose().dot(X),2).sum(axis=1)
        
        logdetA = 2*sum(np.log(np.diag(U))) #
        llh[iter] = 0.5*(d*np.log(alpha)+n*np.log(beta)-alpha*m2-beta*e2-logdetA-n*np.log(2*np.pi))
        if(np.abs(llh[iter]-llh[iter-1]) < tol*np.abs(llh[iter-1])):
            break
        
        V = scipy.linalg.inv(U)
        trS = np.multiply(V,V).sum() # A = inv(S)
        alpha = d/(m2+trS)
        
        UX = mldivide(U.transpose(),X)
        trXSX = np.multiply(UX,UX).sum()
        beta = n/(e2+trXSX)
        
    w0 = tbar - np.multiply(m,xbar).sum(axis=0)
    
    llh = llh[1:iter]
    model.w0 = w0
    model.w = m
    # optional for bayesian probabilistic inference purpose
    model.alpha = alpha
    model.beta = beta
    model.xbar = xbar
    model.U = U
    return model,llh