# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:37:14 2018

@author: SCUTCYF

Naive Bayes with independent Bernoulli

Naive bayes classifier with indepenet Bernoulli.
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
    __slots__ = ["mu","w"]
    def __init__(self,mu,w):
        self.mu = mu
        self.w = w


def nbBern(X, t):
    k = np.max(t)
    n = np.size(X,1)

    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    
    E = csr_matrix((data, ((t-1).reshape(idx.shape),idx)),shape=(k,n)).toarray()
    nk = np.sum(E,1)
    w = nk/n
    mu = (csr_matrix(X)*(E.transpose())*(spdiags(1/nk,0,k,k)))
    
    model.mu = mu # d*k means
    model.w = w
    
    return model
    
    