# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:35:47 2018

@author: SCUTCYF

Accurately compute y = log(1+exp(x))
reference: Accurately Computing log(1-exp(|a|)) Martin Machler

"""
import numpy as np
def log1pexp(x):
    seed = 33.3
    y = x.flatten()
    t = x.flatten()
    length = np.shape(y)[0]
    # idx = x < seed
    for i in range(length):
        if(t[i]<seed):
            y[i] = np.log1p(np.exp(t[i]))
            
    y = np.reshape(y,x.shape)
    return y
