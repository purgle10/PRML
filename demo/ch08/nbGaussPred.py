# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:37:10 2018

@author: SCUTCYF

Prediction of naive Bayes classifier with independent Gaussian.
input:
  model: trained model structure
  X: d x n data matrix
output:
  y: 1 x n predicted class label

"""

import numpy as np

def nbGaussPred(model, X):
    mu = model.mu
    var = model.var
    w = model.w
    assert(np.all(mu.shape==var.shape))
    d = np.size(mu,0)
    
    Lambda = 1/var
    ml = mu*Lambda
    d1 = Lambda.transpose().dot(X*X)-2*ml.transpose().dot(X)
    d2 = np.multiply(mu,ml).sum(axis=0)
    
    M = d1+np.matlib.repmat(d2,np.size(d1,1),1).transpose() # M distance
    c = d*np.log(2*np.pi)+2*np.sum(np.log(var),0).transpose() # normalization constant
    R = -0.5*(M+np.matlib.repmat(c,np.size(M,1),1).transpose())
    y = np.argmax(np.multiply(np.exp(R), np.matlib.repmat(w,np.size(R,1),1).transpose()),0)
    return y
    