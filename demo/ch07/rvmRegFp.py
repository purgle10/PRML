# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:51:56 2018

@author: SCUTCYF

Relevance Vector Machine (ARD sparse prior) for regression
training by empirical bayesian (type II ML) using Mackay fix point update.
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
sys.path.append("..\..\common")
from mldivide import mldivide

class model(object):
    __slots__ = ["index","w0","w","alpha","beta","xbar","U"]
    def __init__(self,index,w0,w,alpha,beta,xbar,U):
        self.index = index
        self.w0 = w0
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.xbar = xbar
        self.U = U

def rvmRegFp(X, t, alpha=0.02, beta=0.5):
    (d,n)=X.shape
    xbar = np.mean(X,1).reshape((X.shape[0],1))
    tbar = np.mean(t,1).reshape((t.shape[0],1))
    X = X-xbar
    t = t-tbar
    XX = X.dot(X.transpose())
    Xt = X.dot(t.transpose())
    
    alpha = np.array([alpha])*np.ones((d,))
    beta = np.array([beta])
    tol = 1e-3
    maxiter = 500
    inf = 1000000
    llh = -inf*np.ones(maxiter)
    index = np.array(range(0,d))
    for iter in range(1,maxiter):
        # remove zeros
        nz = (1/alpha) > tol
        nz = np.nonzero(nz)
        index = index[nz]
        alpha = alpha[nz]
        XXa = XX[nz[0],:]
        XX = XXa[:,nz[0]]
        Xt = Xt[nz]
        X = X[nz]
        
        U = scipy.linalg.cholesky(beta*XX+np.diag(alpha))
        m = beta*mldivide(U,mldivide(U.transpose(),X.dot(t.transpose())))
        m2 = (m*m).reshape(alpha.shape)
        e = np.sum((t-np.transpose(m).dot(X))*(t-np.transpose(m).dot(X)))
        
        logdetS = 2*np.sum(np.log(np.diag(U)))
        llh[iter] = 0.5*(np.sum(np.log(alpha))+n*np.log(beta)-beta*e-logdetS-np.multiply(alpha,m2).sum()-n*np.log(2*np.pi))
        if (np.abs(llh[iter]-llh[iter-1])<tol*np.abs(llh[iter-1])):
            break

        V=np.linalg.inv(U)
        dgSigma = np.multiply(V,V).sum(axis=1).reshape(alpha.shape)
        gamma = 1-alpha*dgSigma
        alpha = gamma/m2
        alpha = alpha.reshape(alpha.shape[0],)
        beta = (n-np.sum(gamma))/e
    
    llh = llh[1:iter+1]
    model.index = index
    model.w0 = tbar-np.multiply(m,xbar[nz]).sum()
    model.w = m
    model.alpha = alpha
    model.beta = beta
    # optional for bayesian probabilistic prediction purpose
    model.xbar = xbar
    model.U = U
    return model,llh