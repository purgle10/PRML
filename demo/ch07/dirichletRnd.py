# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:29:39 2018

@author: SCUTCYF

Generate samples from a Dirichlet distribution.
Input:
  a: k dimensional vector
  m: k dimensional mean vector
Outpet:
  x: generated sample x~Dir(a,m)


"""
import numpy as np

def dirichletRnd(a,m,nargin = 2):
    if(nargin==2):
        a = a*m
    x = np.random.gamma(shape=a,scale=1) # equal to x = gamrnd(a,1); in matlab
    x = x/x.sum()
    return x