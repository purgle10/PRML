# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:01:35 2018

@author: SCUTCYF

compute entropy z=H(x) of a discrete variavle x.
Input:
x: a integer vectors
Output:
z: entropy z=H(x)
"""
import numpy as np
from scipy.sparse import csr_matrix
#from numel import numel


def entropy(x):
    # n = numel(x)
    n = x.size   
    u,t,x = np.unique(x, return_index = True, return_inverse = True)
    
    # k = numel(u)
    k = u.size  
    idx = np.array(range(0,n))
    data = np.ones(idx.shape)
    Mx = csr_matrix((data, (idx,x)), shape=(n,k)).toarray()
    Px = np.mean(Mx,0)
    Hx = -np.dot(Px[np.nonzero(Px)], np.log2(Px[np.nonzero(Px)]))
    z = max(0,Hx)
    return z