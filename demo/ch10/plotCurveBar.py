# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:24:22 2018

@author: SCUTCYF

Plot 1d curve and variance
Input:
  x: 1 x n 
  y: 1 x n 
  sigma: 1 x n or scaler 
"""
import matplotlib.pyplot as plt
import numpy as np


def plotCurveBar(x, y, sigma):
    mycolor = [255/255, 228/255, 225/255]  # pink
    xx = x
    if xx.ndim ==2:
        x = np.sort(xx, axis = 1)
        idx = xx.argsort(axis = 1)
    elif xx.ndim ==1:
        x = np.sort(xx, axis = 0)
        idx = xx.argsort(axis = 0)
    sz = np.shape(sigma)[1] # change 0 to 1, has restriction to input
    idx = np.reshape(idx,(sz,)) # if x and y are vectors, otherwise it will have some problems
    y = y[np.ix_([True],idx)]
    sigma = sigma[np.ix_([True],idx)]
    # sigma = sigma.reshape((1,sz))

    y = np.array(y)
    x = x.reshape(y.shape)
    xf = np.concatenate((x, np.fliplr(x)),1)
    yf = np.concatenate((y+sigma, np.fliplr(y-sigma)),1)
    
    
    plt.fill(xf.flatten(),yf.flatten(),color = mycolor)
    plt.hold(True)
    x = x.reshape((sz,))
    y = y.reshape((sz,))
    plt.plot(x,y,"r-")
    plt.hold(False)
    # v = [x[0],x[-1],-np.inf,np.inf]
    plt.axis()
    # plt.show()