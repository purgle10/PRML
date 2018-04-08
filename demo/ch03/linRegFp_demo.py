# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:02:48 2018

@author: SCUTCYF

demos for ch03
"""
import sys
sys.path.append("../../chapter03")
from linRnd import linRnd 
from linRegFp import linRegFp
from linRegPred import linRegPred
from plotCurveBar import plotCurveBar
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

d = 1
n = 200
x,t = linRnd(d,n)

# Empirical Bayesian linear regression via Mackay fix point iteration method
#fig = plt.figure()
model,llh = linRegFp(x,t)
plt.plot(llh)
y,sigma = linRegPred(model,x,t)
plotCurveBar(x,y,sigma)

mycolor = [255/255,228/255,225/255] # pink
plt.hold(True)
plt.plot(x,t,'o',markerfacecolor='none',markeredgecolor=mycolor)
plt.hold(False)

#pp = PdfPages('plot1.pdf')
#pp.savefig(fig)
#pp.close()