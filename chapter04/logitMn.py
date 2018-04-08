# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:02:30 2018

@author: SCUTCYF

Multinomial regression for multiclass problem (Multinomial likelihood)
Input:
  X: d x n data matrix
  t: 1 x n label (1~k)
  lambda: regularization parameter
Output:
  model: trained model structure
  llh: loglikelihood

"""
import numpy as np
from scipy.sparse import csr_matrix
import sys
sys.path.append("../../common")
from mldivide import mldivide
from logsumexp import logsumexp

class model(object):
    __slots__ = ["W"]
    def __init__(self,W):
        self.W = W
        
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    if ind.any() < 0:
        ind = -1
    elif (ind.any() >= array_shape[0]*array_shape[1]):
        ind = -1
    return ind

def newtonRaphson(X, t, Lambda):
    (d,n) = np.shape(X)
    k = np.max(t)
    tol = 1e-4
    maxiter = 100
    inf = 1000000
    llh = np.ones(maxiter)*(-inf)
    dk = d*k
    idx = np.array(range(0,dk))
    if len(idx) == 0:
        idx = 0
    dg = sub2ind((dk,dk),idx,idx)
    T = csr_matrix((np.ones(n,),((t-1).reshape(n,),np.array(range(0,n)))),shape=(k,n)).toarray()
    W = np.zeros((d,k))
    HT = np.zeros((d,k,d,k))
    for iter in range(1,maxiter):
        A = W.transpose().dot(X)
        logY = A-logsumexp(A,0)
        llh[iter] = np.multiply(T,logY).sum()-0.5*Lambda*np.multiply(W,W).sum()
        if(llh[iter]-llh[iter-1] < tol):
            break
        Y = np.exp(logY)
        for i in range(0,k):
            for j in range(0,k):
                r = Y[i,]*((i==j)-Y[j,]) # r has negative value, so cannot use sqrt
                HT[:,i,:,j]=(X*r).dot(X.transpose())
        G=X.dot((Y-T).transpose())+Lambda*W
        H = np.reshape(HT,(dk,dk))
        Hi = H.flatten()
        Hi[dg] = Hi[dg]+Lambda
        H = Hi.reshape(H.shape)
        Wi = W.flatten()-mldivide(H,G.flatten())
        W = Wi.reshape(W.shape)
        
    llh = llh[1:iter]
    return W,llh
            
def logitMn(X, t, Lambda=1e-4):
    (d,n) = np.shape(X)
    X = np.concatenate((X,np.ones((1,n))),0)
    W, llh = newtonRaphson(X, t, Lambda)
    model.W = W
    return model,llh