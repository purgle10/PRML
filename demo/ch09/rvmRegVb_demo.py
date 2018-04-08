# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:15:29 2018

@author: SCUTCYF
"""

import numpy as np

d = 100
beta = 1e-1
X = np.random.rand(1,d)
w = np.random.randn
b = np.random.randn
t = w.transpose()*X+beta*np.random.randn(1,d)
x = np.linspace(np.min(X),np.max(X),d) # test data

