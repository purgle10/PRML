# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:17:05 2018

@author: SCUTCYF

Plot 2d/3d samples of different classes with different colors.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotClass(X, label=1,nargin=2):
    (d,n) = np.shape(X)
    if nargin==1:
        label = np.ones((n,))
    assert(n==label.size)
    
    mycolor = ['b', 'r', 'g','m','c','y','k']
    c = np.max(label)
    
    fig = plt.gcf()
    if(d==2):
        for i in range(0,c+1):
            idc = label==i
            idc = idc.flatten()
            for j in range(0,np.shape(idc)[0]):
                if(idc[j]):
                    plt.scatter(X[0,j],X[1,j],36,mycolor[i])
    
    elif(d==3):
        ax = Axes3D(fig)
        for i in range(0,c):
            idc = label==i
            idc = idc.flatten()
            for j in range(0,np.shape(idc)[0]):
                if(idc[j]):
                    ax.scatter(X[0,j],X[1,j],X[2,j],36,mycolor[i])
                    
    else:
        return print('ERROR: only support data of 2D or 3D.')
    
    plt.axis('equal')
    plt.grid(True)