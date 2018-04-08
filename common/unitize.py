# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:20:18 2018

@author: SCUTCYF

Unitize the vectors to be unit length
  By default dim = 0 (columns).

"""
import numpy as np

def unitize(X,dim,nargin=2):
    if nargin==1:
        # Determine which dimension sum will use
        dim = np.nonzero(X.shape !=1)
        if dim.size==0:
            dim = 1
        else:
            dim = dim[0]
    s = np.sqrt(np.multiply(X,X).sum(axis=dim))
    Y = X*(1/s)
    return Y,s
