# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:00:05 2018

@author: SCUTCYF

Prediction of naive Bayes classifier with independent Bernoulli.
input:
  model: trained model structure
  X: d x n data matrix
output:
  y: 1 x n predicted class label

"""

import numpy as np
from scipy.sparse import csr_matrix

def nbBernPred(model, X):
    mu = model.mu
    w = model.w
    X = csr_matrix(X)
    R = np.log(mu).transpose()*X+np.log(1-mu).transpose().dot(1-X.toarray())
    R = R+np.matlib.repmat(np.log(w),np.size(R,1),1).transpose()
    # y = np.max(R,0)
    y = np.argmax(R,0)
    return y