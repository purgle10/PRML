# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:15:05 2018

@author: SCUTCYF

Perform EM algorithm for fitting the Gaussian mixture model.
Input: 
  X: d x n data matrix
  init: k (1 x 1) number of components or label (1 x n, 1<=label(i)<=k) or model structure
Output:
  label: 1 x n cluster label
  model: trained model structure
  llh: loglikelihood

"""

import numpy as np
import inspect
from mldivide import mldivide
from logsumexp import logsumexp
from scipy.sparse import csr_matrix
import scipy

class model(object):
    __slots__ = ["mu","Sigma","weight"]
    def __init__(self,mu,Sigma,w):
        self.mu = mu
        self.Sigma = Sigma
        self.w = w

def loggausspdf(X, mu, Sigma):
    d = np.size(X,0)
    X = X - np.matlib.repmat(mu,X.shape[1],1).transpose()
    U = scipy.linalg.cholesky(Sigma)
    Q = mldivide(U.transpose(),X)
    q = np.multiply(Q,Q).sum(axis=0) # quadratic term (M distance)
    c = d*np.log(2*np.pi+2*np.sum(np.log(np.diag(U)))) # normalization constant
    y = -(c+q)/2
    return y

def expectation(X, model):
    mu = model.mu
    Sigma = model.Sigma
    w = model.w
    
    n = np.size(X,1)
    k = np.size(mu,1)
    R = np.zeros((n,k))
    for i in range(0,k):
        R[:,i] = loggausspdf(X, mu[:,i], Sigma[:,:,i])
        
    R = R+np.matlib.repmat(np.log(w),R.shape[1],1).transpose()
    T = logsumexp(R,1)
    llh = np.sum(T)/n # loglikelihood
    R = np.exp(R - np.matlib.repmat(T,R.shape[1],1).transpose())
    return R, llh

def initialization(X, init):
    n = np.size(X,1)
    if inspect.isclass(init): # init with a model
        R = expectation(X, init)
        return R
    elif int == type(init):
        k = init
        label = np.ceil(k*np.random.rand(1,n))
        R = csr_matrix((np.ones(label.shape[0],),(np.array(range(0,n)),(label).reshape(label.shape[0],))),shape=(n,k)).toarray()
        return R
    elif np.all(np.size(init) == (n)): # init with labels
        label = init
        k = np.max(label)
        idx = np.array(range(0,n))
        R = csr_matrix((np.ones(idx.shape),(idx,label.reshape(idx.shape))),shape=(n,k)).toarray()
        return R
    else:
        print("ERROR: init is not valid.")
    

def maximization(X, R):
    (d,n) = X.shape
    k = np.size(R,1)
    nk = np.sum(R,0)
    w = nk/n
    mu = X.dot(R).dot(1/nk)
    
    Sigma = np.zeros((d,d,k))
    r = np.sqrt(R)
    for i in range(0,k):
        Xo = X - mu[:,i]
        Xo = Xo*(r[:,i].transpose())
        Sigma[:,:,i] = Xo*Xo.transpose()/nk[i]+np.eye(d)*(1e-6)
        
    model.mu = mu
    model.Sigma = Sigma
    model.w = w

def mixGaussEm(X, init):
    print("EM for Gaussian mixture: running ... \n")
    tol = 1e-6
    maxiter = 500
    inf = -10000000
    llh = -inf*np.ones((maxiter+1,1))
    R = initialization(X, init)
    for iter in range(1,maxiter):
        label = np.argmax(R, 1)
        uniIdx = np.unique(label)
        R = R[:,uniIdx]
        model = maximization(X, R)
        R,llh[iter] = expectation(X, model)
        if(np.abs(llh[iter]-llh[iter-1]) < tol*np.abs(llh[iter-1])):
            break
        
    llh = llh[1:iter]
    return label,model,llh
        