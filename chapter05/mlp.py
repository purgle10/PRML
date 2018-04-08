# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:11:12 2018

@author: SCUTCYF

Multilayer perceptron
Input:
  X: d x n data matrix
  Y: p x n response matrix
  h: L x 1 vector specify number of hidden nodes in each layer l
Ouput:
  model: model structure
  mse: mean square error

"""

import numpy as np
import sys
sys.path.append("../../common")
from log1pexp import log1pexp

class model(object):
    __slots__ = ["W"]
    def __init__(self,W):
        self.W = W

def mlp(X, Y, h):
    h1 = [X.shape[0]]
    h2 = [Y.shape[0]]
    h = np.append(h1,h.tolist()+h2)
    L = h.size
    W = np.zeros(((L-1)*(L-1),),dtype=np.object)
    for l in range(0,L-1):
        W[l] = np.random.randn(h[l],h[l+1])
    Z = np.zeros((L*L,),dtype=np.object)
    Z[0] = X
    eta = 1.0/X.shape[1]
    maxiter = 2000
    mse = np.zeros((maxiter,))
    for iter in range(0,maxiter):
        # forward
        for l in range(1,L):
            Z[l] = np.exp(-log1pexp(-np.transpose(W[l-1]).dot(Z[l-1]))) # Z{l} = sigmoid(W{l-1}'*Z{l-1}); in matlab
        
        # backward
        E = Y-Z[L-1]
        mse[iter] = np.mean(np.multiply(E,E).sum())
        for l in range(L-2,-1,-1):
            df = Z[l+1]*(1-Z[l+1])
            dG = df*E
            dW = Z[l].dot(dG.transpose())
            W[l] = W[l] + eta*dW
            E = W[l].dot(dG)
        
    mse = mse[0:iter]
    model.W = W
    return model,mse
            