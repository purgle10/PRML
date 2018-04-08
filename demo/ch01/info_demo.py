# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:01:35 2018

@author: SCUTCYF
"""
import sys
sys.path.append("../../chapter01")
import numpy as np
from entropy import entropy
from jointEntropy import jointEntropy
from condEntropy import condEntropy
from mutInfo import mutInfo
from relatEntropy import relatEntropy
from nmi import nmi
from nvi import nvi

sys.path.append("../../common")
from isequalf import isequalf
k = 10  # variable range
n = 100  # number of variables
x = np.ceil(k*np.random.rand(1, n))
y = np.ceil(k*np.random.rand(1, n))

# Entropy H(x), H(y)
Hx = entropy(x)
Hy = entropy(y)

# Joint entropy H(x,y)
Hxy = jointEntropy(x,y)

# Conditional entropy
Hx_y = condEntropy(x,y)

# Mutual information I(x,y)
Ixy = mutInfo(x,y)

# Relative entropy (KL divergence) KL(p(x)|p(y))
Dxy = relatEntropy(x,y)

# Normalized mutual information I_n(x,y)
nIxy = nmi(x,y)

# Nomalized variation information I_v(x,y)
vIxy = nvi(x,y)

# H(x|y) = H(x,y)-H(y)
print(isequalf(Hx_y,Hxy-Hy))

# I(x,y) = H(x)-H(x|y)
print(isequalf(Ixy,Hx-Hx_y))

# I(x,y) = H(x)+H(y)-H(x,y)
print(isequalf(Ixy,Hx+Hy-Hxy))

# I_n(x,y) = I(x,y)/sqrt(H(x)*H(y))
print(isequalf(nIxy,Ixy/np.sqrt(Hx*Hy)))

# I_v(x,y) = (1-I(x,y)/H(x,y))
print(isequalf(vIxy,1-Ixy/Hxy))