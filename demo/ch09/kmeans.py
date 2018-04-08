# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:20:40 2018

@author: SCUTCYF

Perform k-means clustering.
Input:
  X: d x n data matrix
  init: k number of clusters or label (1 x n vector)
Output:
  label: 1 x n cluster label
  energy: optimization target value
  model: trained model structure

"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

def kmeans(X, init):
    init = np.array(init)
    n = np.size(X,1)
    if init.size == 1:
        k = init
        label = np.ceil(k*np.random.rand(1,n)) # random initial label
    elif init.size == n:
        label = init
        
    last = 0
    
    while(np.any(label!=last)):
        u, tmp, label[:] = np.unique(label, return_index=True,return_inverse=True) # remove empty cluster
        k = u.size
        
        idx = np.array(range(0,n))
        data = np.ones(idx.shape)
    
        E = csr_matrix((data, (idx,label.reshape(idx.shape))),shape=(n,k)).toarray() # transform label into indices
        m = X.dot(E*(spdiags(1/np.sum(E,0).transpose(),0,k,k))) # compute centers
        last = label
        m1 = m.transpose().dot(X)
        m2 = np.multiply(m,m).sum(axis=0).transpose()/2
        m3 = m1-np.matlib.repmat(m2,np.size(m1,1),1).transpose()
        val = np.max(m3,0)
        label = np.argmax(m3,0)
        
    energy = np.multiply(X[:],X[:]).sum()-2*np.sum(val)
    model_means = m
    return label,energy,model_means
        