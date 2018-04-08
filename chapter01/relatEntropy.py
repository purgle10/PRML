# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:49:13 2018

@author: SCUTCYF
Compute relative entropy (a.k.a KL divergence) z=KL(p(x)||p(y)) of two discrete variables x and y.
Input:
  x, y: two integer vector of the same length 
Output:
  z: relative entropy (a.k.a KL divergence) z=KL(p(x)||p(y))
"""
# from numel import numel
import numpy as np
from scipy.sparse import csr_matrix
def relatEntropy(x,y):
    assert(x.size==y.size)
    n = x.size
    x = np.reshape(x,(n,))
    y = np.reshape(y,(n,))
    
    l = min(np.min(x),np.min(y))
    x = x-l+1
    y = y-l+1
    k = max(np.max(x),np.max(y))
    
    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    
    Mx = csr_matrix((data, (idx,x-1)),shape=(n,k))
    My = csr_matrix((data, (idx,y-1)),shape=(n,k))
    
    Mx = np.mean(Mx.toarray(),0)
    My = np.mean(My.toarray(),0)
    
    Px = np.nonzero(Mx)
    Py = np.nonzero(My)
    
    z = -np.dot(Mx[Px],np.log2(My[Py])-np.log2(Mx[Px]))
    z = max(0,z)
    return z
