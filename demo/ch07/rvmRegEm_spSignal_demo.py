# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:54:11 2018

@author: SCUTCYF

sparse signal recovery demo

"""
import sys
sys.path.append("..\..\common")
from unitize import unitize
import numpy as np
from rvmRegEm import rvmRegEm
import matplotlib.pyplot as plt

d = 512 # signal length
k = 20 # number of spikes
n = 100 # number of mearsurements

# random +/- 1 signal
x = np.zeros((d,1))
q = np.random.permutation(d)
x[q[0:k]] = np.sign(np.random.randn(k,1))

# projection matrix
A= unitize(np.random.randn(d,n),0)[0]
# noisy observations
sigma = 0.005
e = sigma*np.random.randn(1,n)
y = x.transpose().dot(A)+e

model,llh = rvmRegEm(A,y)
plt.plot(llh)

m = np.zeros((d,1))
m[model.index] = model.w

h = np.max(np.abs(x))+0.2
# x_range = np.array(range(1,d+1))
# y_range = np.array(range(-h,+h+1))
v = [1,d,-h,h]
plt.figure()
plt.subplot(2,1,1)
plt.plot(x)
plt.axis(v)
plt.title('Original Signal')
plt.subplot(2,1,2)
plt.plot(m)
plt.axis(v)
plt.title('Restored Signal')