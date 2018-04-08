# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:33:08 2018

@author: SCUTCYF
Compute conditional entropy z=H(x|y) of two discrete variables x and y.
Input:
  x, y: two integer vector of the same length 
Output:
  z: conditional entropy z=H(x|y)
"""
import numpy as np
# from numel import numel
from scipy.sparse import csr_matrix
def condEntropy(x,y):
    assert(x.size == y.size)
    # n = numel(x)
    n = x.size
    x = np.reshape(x, (n,))
    y = np.reshape(y, (n,))
    
    l = min(np.min(x),np.min(y))
    x = x-l+1
    y = y-l+1
    k = max(np.max(x),np.max(y))
    
    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    
    Mx = csr_matrix((data, (idx,x-1)),shape=(n,k))
    My = csr_matrix((data, (idx,y-1)),shape=(n,k))
    Pxy = Mx.transpose()*My/n # joint distribution
    Pxy = Pxy.toarray().flatten()
    Hxy = -np.dot(Pxy[np.nonzero(Pxy)], np.log2(Pxy[np.nonzero(Pxy)]))
    
    Py = np.mean(My.toarray(), 0)
    Hy = -np.dot(Py[np.nonzero(Py)],np.log2(Py[np.nonzero(Py)]))
    
    # conditional entropy H(x|y)
    z = Hxy-Hy
    z = max(0,z)
    return z