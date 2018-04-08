# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:31:09 2018

@author: SCUTCYF

Naive bayes classifier with indepenet Gaussian, each dimension of data is
assumed from a 1d Gaussian distribution with independent mean and variance.
Input:
  X: d x n data matrix
  t: 1 x n label (1~k)
Output:
  model: trained model structure

"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

class model(object):
    __slots__ = ["mu","var","w"]
    def __init__(self,mu,var,w):
        self.mu = mu
        self.var = var
        self.w = w
        
def nbGauss(X, t):
    k = np.max(t)
    n = np.size(X,1)
    
    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    
    E = csr_matrix((data, ((t-1).reshape(idx.shape),idx)),shape=(k,n)).toarray()
    nk = np.sum(E,1)
    w = nk/n
    R = (E.transpose())*(spdiags(1/nk,0,k,k))
    mu = X.dot(R)
    var = np.multiply(X,X).dot(R)-mu*mu
    
    model.mu = mu # d * k means
    model.var = var # d * k variances
    model.w = w
    return model