# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:30:14 2018

@author: SCUTCYF

Generate samples from a Gaussian distribution.
Input:
  mu: d x 1 mean vector
  Sigma: d x d covariance matrix
  n: number of samples
Outpet:
  x: d x n generated sample x~Gauss(mu,Sigma)

"""

import numpy as np
import scipy

def gaussRnd(mu, Sigma, n=1):
    V = scipy.linalg.cholesky(Sigma)
    VV = V.transpose().dot(np.random.randn(np.size(V,0),n))
    x = VV+np.matlib.repmat(mu,n,1).transpose().reshape(VV.shape)
    return x