# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:19:51 2018

@author: SCUTCYF

Genarate samples form a Gaussian mixture model.
Input:
  d: dimension of data
  k: number of components
  n: number of data
Output:
  X: d x n data matrix
  z: 1 x n response variable
  model: model structure

"""
import numpy as np
from dirichletRnd import dirichletRnd
from discreteRnd import discreteRnd
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from gaussRnd import gaussRnd

class model(object):
    __slots__ = ["mu","Sigma","weight"]
    def __init__(self,mu,Sigma,weight):
        self.mu = mu
        self.Sigma = Sigma
        self.weight = weight


def invwishartrand(nu, phi):
    return inv(wishartrand(nu, inv(phi)))

def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def mixGaussRnd(d,k,n):
    alpha0 = 1 # hyperparameter of Dirichlet prior
    W0 = np.eye(d) # hyperparameter of inverse Wishart prior of covariances
    v0 = d+1 # hyperparameter of inverse Wishart prior of covariances
    mu0 = np.zeros((d,1)) # hyperparameter of Guassian prior of means
    beta0 = np.power(k,1.0/d) # hyperparameter of Guassian prior of means % in volume x^d there is k points: x^d=k
    
    w = dirichletRnd(alpha0, np.ones((1,k))/k)
    z = discreteRnd(w,n)
    
    mu = np.zeros((d,k))
    Sigma = np.zeros((d,d,k))
    X = np.zeros((d,n))
    for i in range(0,k):
        idx = z-1==i
        Sigma[:,:,i] = invwishartrand(v0,W0)
        mu[:,i] = gaussRnd(mu0,beta0*(Sigma[:,:,i])).reshape(mu[:,i].shape)
        X[:,np.nonzero(idx)] = gaussRnd(mu[:,i],Sigma[:,:,i],np.sum(idx))
        
    model.mu = mu
    model.Sigma = Sigma
    model.weight = w
    
    return X,z,model