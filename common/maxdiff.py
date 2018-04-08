# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:44:07 2018

@author: SCUTCYF
"""
import numpy as np

def maxdiff(x,y):
    assert(np.all(x.shape==y.shape))
    z = np.max(np.abs(x.flatten()-y.flatten()))
    return z