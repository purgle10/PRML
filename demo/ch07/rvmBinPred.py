# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:38:24 2018

@author: SCUTCYF

Prodict the label for binary logistic regression model
Input:
  model: trained model structure
  X: d x n testing data
Output:
  y: 1 x n predict label (0/1)
  p: 1 x n predict probability [0,1]

"""

import numpy as np
import sys
sys.path.append("..\..\common")
from log1pexp import log1pexp

def rvmBinPred(model, X):
    index = model.index
    (d,n) = np.shape(X)
    X = np.concatenate((X,np.ones((1,n))),0)
    X = X[index]
    w = model.w
    A = w.transpose().dot(X)
    p = np.exp(-log1pexp(A))
    y =  np.round(p)
    return y,p
