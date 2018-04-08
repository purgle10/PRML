# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:36:49 2018

@author: SCUTCYF

alternative to mldivide,“\” matlab operator
"""

from itertools import combinations
import numpy as np

def mldivide(A,b):
    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    if rank == num_vars:              
        sol = np.linalg.lstsq(A, b)[0]    # not under-determined
        return sol
    else:
        for nz in combinations(range(num_vars), rank):    # the variables not set to zero
            try: 
                sol = np.zeros((num_vars, 1))  
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
                return sol
            except np.linalg.LinAlgError:     
                pass                    # picked bad variables, can't solve
