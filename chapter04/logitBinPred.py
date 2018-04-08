# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:58:56 2018

@author: SCUTCYF

Prediction of binary logistic regression model
Input:
  model: trained model structure
  X: d x n testing data
Output:
  y: 1 x n predict label (0/1)
  p: 1 x n predict probability [0,1]


"""
import numpy as np
from log1pexp import log1pexp
def logitBinPred(model, X):
    (d,n) = np.shape(X)
    X = np.concatenate((X,np.ones((1,n))),0)
    w = model.w
    p = np.exp(-log1pexp(-w.transpose().dot(X)))
    y = np.round(p)
    return y,p