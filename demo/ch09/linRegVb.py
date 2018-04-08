# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:21:01 2018

@author: SCUTCYF

Variational Bayesian inference for linear regression.
Input:
  X: d x n data
  t: 1 x n response
  prior: prior parameter
Output:
  model: trained model structure
  energy: variational lower bound

"""

def linRegVb(X, t, prior):
    