# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:09:00 2018

@author: SCUTCYF

Empirical Bayesian linear regression via EM

"""

import matplotlib.pyplot as plt
from linRnd import linRnd
from linRegEm import linRegEm
d = 5;
n = 200;
[x,t] = linRnd(d,n)
[model,llh] = linRegEm(x,t)
plt.plot(llh)