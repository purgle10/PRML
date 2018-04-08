# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:23:44 2018

@author: SCUTCYF

Generate samples from a Gaussian mixture distribution with common variances (kmeans model).
Input:
  d: dimension of data
  k: number of components
  n: number of data
Output:
  X: d x n data matrix
  z: 1 x n response variable
  mu: d x k centers of clusters
  
"""
import numpy as np
from dirichletRnd import dirichletRnd
from discreteRnd import discreteRnd
from scipy.sparse import csr_matrix

def kmeansRnd(d,k,n):
    alpha = np.array([1])
    beta = np.power(k,1.0/d) # in volume x^d there is k points: x^d=k
    
    X = np.random.randn(d,n)
    w = dirichletRnd(alpha,np.ones((1,k))/k)
    z = discreteRnd(w,n)
    E = csr_matrix((np.ones(z.shape[1],),((z-1).reshape(z.shape[1],),np.array(range(0,n)))),shape=(k,n)).toarray()
    mu = np.random.randn(d,k)*beta
    X = X+mu.dot(E)
    return X,z,mu
