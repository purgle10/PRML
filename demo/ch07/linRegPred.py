# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:58:54 2018

@author: SCUTCYF

Compute linear regression model reponse y = w'*X+w0 and likelihood
Input:
  model: trained model structure
  X: d x n testing data
  t (optional): 1 x n testing response
Output:
  y: 1 x n prediction
  sigma: variance
  p: 1 x n likelihood of t
"""
import numpy as np
import sys
sys.path.append("..\..\common")
sys.path.append("..\../chapter03")
from mldivide import mldivide
from logGauss import logGauss

def linRegPred(model, X, t, nargout = 2):
    w = model.w
    w0 = model.w0
    y = np.transpose(w)*X+w0 # X and w already are matrix type
    
    # probability prediction
    beta = model.beta
    if(hasattr(model,"U")): # python3 will have no problem, but python2 is dangerous to use hasattr()
        U = model.U
        Xo = X - model.xbar
        if U.size==1:
            XU = Xo/U # equals to U\Xo in matlab if U is 1*1 scalar
        else:
            XU = mldivide(np.transpose(U), Xo)
        sigma = np.sqrt((1+np.multiply(XU,XU).sum(axis=0))/beta)
    else:
        sigma = np.sqrt(1/beta)*np.ones((1,np.size(X,1)))
#    try:
#        U = model.U

#    except AttributeError:
#        sigma = np.sqrt(1/beta)*np.ones((1,np.size(X,1)))
#        print("no U!")
    if nargout == 3:
        p = np.exp(logGauss(t,y,sigma))
        return y,sigma, p
    
    return y,sigma