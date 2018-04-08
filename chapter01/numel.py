# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:04:47 2018

@author: SCUTCYF
"""

def numel(x):
    size = x.shape
    length = len(size)
    assert(length!=0)
    if length==1:
        return size[0]
    n = 1
    for i in range(length): #
        n = n*size[i]
    return n