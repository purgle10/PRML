# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:06:38 2018

@author: SCUTCYF

demo for knCenter

"""
import sys
sys.path.append("../../chapter06")
import numpy as np
sys.path.append("../../common")
from maxdiff import maxdiff
# import knGauss.knGauss as kn
from knCenter import knCenter

X = np.random.rand(2,100)
X1 = np.random.rand(2,10)
X2 = np.random.rand(2,5)

aa = maxdiff(knCenter(2,X,X1),np.array([np.diag(knCenter(3,X,X1,X1))]))
print(aa)
bb = maxdiff(knCenter(1,X),knCenter(3,X,X,X))
print(bb)
