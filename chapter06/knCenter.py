# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:32:22 2018

@author: SCUTCYF

Centerize the data in the kernel space
Input:
  kn: kernel function
  X: d x n data matrix of which the center in the kernel space is computed
  X1, X2: d x n1 and d x n2 data matrix. the kernel k(x1,x2) is computed
      where the origin of the kernel space is the center of phi(X)
Ouput:
  Kc: n1 x n2 kernel matrix between X1 and X2 in kernel space centered by
      center of phi(X)

"""
import numpy as np
from knGauss import knGauss as kn

def knCenter(nargin, X, X1=0, X2=0):
    K = kn(X,X)
    mK = np.mean(K,axis = 0)
    mmK = np.mean(mK)
    if nargin == 1: # compute the pairwise centerized version of the kernel of X. eq knCenter(kn,X,X,X)
        mK1 = mK.reshape((mK.size,1)) # because mK is vector, so mK.size is suitable
        mK2 = mK.reshape((1,mK.size))
        Kc = K+mmK-(mK1+mK2)
        return Kc # Kc = K-M*K-K*M+M*K*M; where M = ones(n,n)/n; 
    elif nargin == 2: # compute the norms (k(x,x)) of X1 w.r.t. the center of X as the origin. eq diag(knCenter(kn,X,X1,X1))
        Kc = kn(X1,0,1,1)+mmK-2*np.mean(kn(X,X1),axis=0)
        return Kc
    elif nargin == 3: # compute the kernel of X1 and X2 w.r.t. the center of X as the origin
        t = np.mean(kn(X,X1),axis=0)
        t = t.reshape((t.size,1))
        s = np.mean(kn(X,X2),axis=0)
        s = s.reshape((1,t.size))
        Kc = kn(X1,X2)+mmK-(s+t)
        return Kc