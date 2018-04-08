# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:07:33 2018

@author: SCUTCYF

Prediction of multiclass (multinomial) logistic regression model
Input:
  model: trained model structure
  X: d x n testing data
Output:
  y: 1 x n predict label (1~k)
  P: k x n predict probability for each class

"""
import numpy as np
import sys
sys.path.append("../../common")
from logsumexp import logsumexp

def logitMnPred(model, X):
    W = model.W
    (d,n) = np.shape(X)
    X = np.concatenate((X,np.ones((1,n))),0)
    A = W.transpose().dot(X)
    P = np.exp(A-logsumexp(A,0))
    y =  np.argmax(P,axis=0)
    return y,P
