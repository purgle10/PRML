# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:02:25 2018

@author: SCUTCYF

Logistic logistic regression for multiclass classification

"""
import sys
sys.path.append("../../chapter04")
from kmeansRnd import kmeansRnd
from logitMn import logitMn
from logitMnPred import logitMnPred
from plotClass import plotClass
k = 3
n = 1000
X,t,mu = kmeansRnd(2,k,n)
model,llh = logitMn(X,t)
y,P = logitMnPred(model,X)
plotClass(X,y)