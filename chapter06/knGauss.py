# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:09:46 2018

@author: SCUTCYF

Gaussian (RBF) kernel K = exp(-|x-y|/(2s));
Input:
  X: d x nx data matrix
  Y: d x ny data matrix
  s: sigma of gaussian
Ouput:
  K: nx x ny kernel matrix


"""
import numpy as np

def knGauss(X, Y=0, s=1,nargin=2):
    if nargin < 2:
        K = np.ones((1,X.shape[1]))
        return K
    else:
        X1 = np.multiply(X,X).sum(axis=0).reshape((X.shape[1],1))
        Y1 = np.multiply(Y,Y).sum(axis=0).reshape((1,Y.shape[1]))
        D = X1+Y1-2*np.transpose(X).dot(Y)
        K = np.exp(D/(-2*s*s))
        return K
