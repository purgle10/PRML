# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:47:54 2018

@author: SCUTCYF

Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
  By default dim = 1 (columns).

"""

import numpy as np

def logsumexp(X, dim=0,nargin=2):
    dimidx = np.nonzero(X.shape)
    if nargin==1:
        if(len(dimidx)):
            dim = dimidx[0]
        else:
            dim = 0
    
    y = np.max(X,dim)
    s = y+np.log(np.sum(np.exp(X-y),dim))
    i = np.isinf(y)
    if(np.any(i.flatten())):
        yi = y.flatten()
        si = s.flatten()
        ii = i.flatten()
        si[ii]=yi[ii]
        s = np.reshape(si,s.shape)
    return s