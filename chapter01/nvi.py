# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:57:56 2018

@author: SCUTCYF
Compute normalized variation information z=(1-I(x,y)/H(x,y)) of two discrete variables x and y.
Input:
  x, y: two integer vector of the same length 
Output:
  z: normalized variation information z=(1-I(x,y)/H(x,y))
"""
# from numel import numel
import numpy as np
from scipy.sparse import csr_matrix
def nvi(x,y):
    assert(x.size==y.size)
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

    # hacking, to elimative the 0log0 issue
    Px = np.mean(Mx.toarray(),0)
    Py = np.mean(My.toarray(),0)
    
    # entropy of Px and Py
    Hx = -np.dot(Px[np.nonzero(Px)], np.log2(Px[np.nonzero(Px)]))
    Hy = -np.dot(Py[np.nonzero(Py)], np.log2(Py[np.nonzero(Py)]))
    
    # normalized variation information
    z = 2 - (Hx+Hy)/Hxy
    z = max(0,z)
    return z