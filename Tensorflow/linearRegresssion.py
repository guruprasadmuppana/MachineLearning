# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:44:03 2019

@author: Admin
"""

# Linear regression
# Y = W.T + b
# Sigmod function: Z = 1/(1+exp(-Y))

import numpy as np

from tools import sigmod


N = 5
D = 2 

X = np.random.randn(N,D) # N sample with two Dimensional. W0 + W1
b = np.ones((N,1))  # makes column vector with N rows and 1 column. It takes a tuple 

Xb = np.concatenate((X,b),axis=1) # axis = 1 says that you need to operate at the row level; not column level
                                # It means, the number of rows in both matrices should be the same.
                                # axis = 0 acts by column. For example, you want to find the mean of a column of data for all rows.

# Wieghts
W = np.random.randn(D+1) # W = [ W0,W1,W2.. ] , W0 = 1 ; 1x(D+1) dimentions. it is an array

Y = Xb.dot(W) # N*(D+1), ((1*(D+1))T => Nx1 = Y = X0*W0+X1*W1+X2*W2 rows=> dot product.

# Sigmod function: activation function.
# All Z values will between 0 and 1
Z = sigmod(Y)

