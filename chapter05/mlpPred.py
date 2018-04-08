# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:13:36 2018

@author: SCUTCYF

Multilayer perceptron prediction
Input:
  model: model structure
  X: d x n data matrix
Ouput:
  Y: p x n response matrix


"""
import numpy as np
import sys
sys.path.append("../../common")
from log1pexp import log1pexp

def mlpPred(model, X):
    W = model.W
    L = int(np.sqrt(len(W)))+1
    Z = np.zeros((L*L,),dtype=np.object)
    Z[0] = X
    for l in range(1,L):
        Z[l] = np.exp(-log1pexp(-np.transpose(W[l-1]).dot(Z[l-1]))) # Z{l} = sigmoid(W{l-1}'*Z{l-1}); in matlab
    y = Z[L-1]
    return y