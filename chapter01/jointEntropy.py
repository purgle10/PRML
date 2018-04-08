# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:01:35 2018

@author: SCUTCYF

Compute joint entropy z=H(x,y) of two discrete variables x and y.
Input:
  x, y: two integer vector of the same length 
  Output:
  z: joint entroy z=H(x,y)
  
"""
import numpy as np
# from numel import numel
from scipy.sparse import csr_matrix
def jointEntropy(x,y):
    assert(x.size == y.size)
    n = x.size
    x = np.reshape(x, (n,))
    y = np.reshape(y, (n,))
    l = min(np.min(x),np.min(y))
    x = x-l+1
    y = y-l+1
    k = max(np.max(x),np.max(y))
    
    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    pp = csr_matrix((data, (idx,x-1)), shape=(n,k)).transpose()
    P = pp*(csr_matrix((np.ones(y.shape), (idx,y-1)), shape=(n,k))/n) # joint distribution
    P = P.toarray().flatten()
    p = np.nonzero(P)
    z = -np.dot(P[p], np.log2(P[p]))
    z = max(0,z)
    return z