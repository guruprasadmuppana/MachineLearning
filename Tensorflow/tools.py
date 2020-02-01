# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:04:57 2019

@author: Admin
"""

import numpy as np

# Activation functions

# Sigmod function : Y = 1/(1+exp(-Z)); Z is a linear equavation. Y is sigmod function
# sigmod function ranges between (0,1)
def sigmod(Z):
    return 1/(1+np.exp(-Z))


import matplotlib.pyplot as plt

mu, sigma = 0,0.1
samples = np.random.normal(mu,sigma,1000)
#samples = np.random.randn(1000)
print (abs(mu-np.mean(samples))< 0.01 ) # ?
count,bins, ignored = plt.hist(samples, 30, normed=True)
y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(-bins-mu)**2/ (2*sigma**2))
plt.plot(bins,y, linewidth=2, color ="r" )