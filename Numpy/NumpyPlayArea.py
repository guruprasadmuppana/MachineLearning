# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 06:01:48 2019

@author: Guru Prasad

Description : This is a play area for Numpy functions

Reference: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html

"""

# Rand  - Random values in a given shape. All values are between 0 and 1
# Randn - Returns samples from  standard normal distribution with mu = 0 and std = 1.

# shuffle - Shuffles in place. No return values.


import numpy as np
import matplotlib.pyplot as plt


shape1 = 100
r,c = 1000,1000
one_d = np.random.rand(shape1)
mul_d = np.random.rand(r,c)
#print(np.random.rand(shape1))
#print(np.random.rand(r,c))

#plt.plot(one_d)
#plt.show()
#
one_d = np.random.randn(shape1)
mul_d = np.random.randn(r,c)
#
#print(np.random.rand(shape1))
#print(np.random.rand(r,c))

#plt.plot(one_d)
#plt.hist(one_d,bins=25)
#plt.show()

one_d = np.arange(shape1) # generates the numbers from 0 to shape1. It is not random but in the order.
np.random.shuffle(one_d)
#print(one_d)

