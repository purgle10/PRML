# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:02:02 2018

@author: SCUTCYF

Determine whether two float number x and y are equal up to precision tol
"""
import numpy as np
def isequalf(x,y,tol = 1e-8):
  
    assert(np.all(np.size(x)==np.size(y)))
    z = max(abs(x.flatten()-y.flatten())) < tol
    return z
