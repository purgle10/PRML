# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:50:39 2018

@author: SCUTCYF

Compute log pdf of a Gaussian distribution.
Input:
  X: d x n data matrix
  mu: d x 1 mean vector of Gaussian
  sigma: d x d covariance matrix of Gaussian
Output: 
  y: 1 x n probability density in logrithm scale y=log p(x)
"""
import numpy as np
import scipy
from mldivide import mldivide

def logGauss(X, mu, sigma):
    (d,k) = np.shape(mu)
    y = np.array([])
    if(np.all(np.array(np.shape(sigma))) and k == 1):
        X = X - mu
        R = scipy.linalg.cholesky(sigma)
        Q = mldivide(np.transpose(R), X)
        q = np.multiply(Q,Q).sum(axis=0) # quadratic term ( M distance) equals to q = dot(Q,Q,1) in matlab
        c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(R))) # normalization constant
        y = -0.5*(c+q)
    elif(np.shape(sigma)[0] == 1 and np.shape(sigma)[1] == np.shape(mu)[1]) : #k mu and (k or one) scalar sigma
        XXdot = np.multiply(X,X).sum(axis=0)
        X2 = np.matlib.repmat(np.transpose(XXdot),1,k)
        D = X2 - 2*np.transpose(X).dot(mu) + np.multiply(mu,mu).sum(axis=0)
        q = D*(1/sigma) # M distance
        c = d*np.log(2*np.pi) + 2*np.log(sigma) # normalization constant
        y = -0.5*(q+c)
        
    return y