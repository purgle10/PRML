# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:50:22 2018

@author: SCUTCYF

Generate samples from a discrete distribution (multinomial).
Input:
  p: k dimensional probability vector
  n: number of samples
Ouput:
  x: k x n generated samples x~Mul(p)

"""
import numpy as np

def discreteRnd(p, n=1):
    r = np.random.rand(1,n)
    pp = np.array(np.cumsum(p.flatten()))
    p = pp/pp[-1]
    aa = p.tolist()
#    aa.reverse()
#    aa.append(0)
#    aa.reverse()
    bb = list([0])+aa
    x = np.digitize(r,np.array(bb))
    return x
