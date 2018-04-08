# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:15:41 2018

@author: SCUTCYF

Perform k-medoids clustering.
Input:
  X: d x n data matrix
  init: k number of clusters or label (1 x n vector)
Output:
  label: 1 x n cluster label
  energy: optimization target value
  index: index of medoids

"""
import numpy as np
from scipy.sparse import csr_matrix

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    if ind.any() < 0:
        ind = -1
    elif (ind.any() >= array_shape[0]*array_shape[1]):
        ind = -1
    return ind

def kmedoids(X, init):
    (d,n) = X.shape
    init = np.array(init)
    if init.size == 1:
        k = init
        label = np.ceil(k*np.random.rand(1,n)) # random initial label
    elif init.size == n:
        label = init
        
    X = X -  np.matlib.repmat(np.mean(X,1),X.shape[1],1).transpose()
    v = np.multiply(X,X).sum(axis = 0)
    D = v.reshape(v.size,1)+v.reshape(1,v.size) - 2*(X.transpose().dot(X))
    
    idx = np.transpose(np.array(range(0,d)))
    D[0,sub2ind((d,d),idx,idx)]=0
        
    last = 0
    
    while(np.any(label!=last)):
        u, tmp, label[:] = np.unique(label, return_index=True,return_inverse=True) # remove empty cluster
        k = u.size
        
        idx = np.array(range(0,n))
        data = np.ones(idx.shape)
    
        E = csr_matrix((data, (idx,label.reshape(idx.shape))),shape=(n,k)).toarray()
        index = np.argmin(D.dot(E),0)
        last = label
        val = np.min(D[index,:],0)
        label = np.argmin(D[index,:],0)
        
    energy = np.sum(val)
    return label,energy,index