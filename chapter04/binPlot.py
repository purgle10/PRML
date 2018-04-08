# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:03:03 2018

@author: SCUTCYF

Plot binary classification result for 2d data
Input:
  model: trained model structure
  X: 2 x n data matrix
  t: 1 x n label


"""

import numpy as np
import matplotlib.pyplot as plt
#from cycler import cycler

def binPlot(model, X, t):
    assert(np.shape(X)[0]==2)
    w = model.w
    xi = np.min(X,1) #
    xa = np.max(X,1) #
    xv = np.linspace(xi[0],xa[0])
    yv = np.linspace(xi[1],xa[1])
    x1,x2 = np.meshgrid(xv,yv)
    
    # mycolor = 'brgmcyk'
    # m = len(mycolor)
    mycolor = ['b', 'r', 'g','m','c','y','k']
    plt.gcf()
    plt.axis('equal')
    plt.clf()
    # plt.hold(True)
    # fig.set_prop_cycle(cycler('color', ['b', 'r', 'g','m','c','y','k']))
    for i in range(0,int(np.max(t+1))):
        idc = (t==i)
        idc = idc.flatten()
        for j in range(0,np.shape(X)[1]):
            if(idc[j]):
                plt.scatter(X[0,j],X[1,j],36,mycolor[i])
        
    y=w[0]*x1+w[1]*x2+w[2]
    plt.contour(x1,x2,y,[0])
    # plt.hold(False)