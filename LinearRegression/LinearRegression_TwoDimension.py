# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:49:34 2020

@author: Guru Prasad Muppana

Y = w1*x1 + w2*x2

Using Ax=B  using linear algebra.


"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

N = 100 # range of x1 and x2.
TotalSample = 1000

# Generate the data points.

w = np.array([2, 3]) # parameters of the plan.
with open('data_2d_plain.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=(TotalSample,2))
    Y = np.dot(X, w) + 1 + np.random.normal(scale=5, size=TotalSample)
    for i in range(TotalSample):
        f.write("%s,%s,%s\n" % (X[i,0], X[i,1], Y[i]))


# load the data points
X = []
Y = []
for line in open('data_2d_plain.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) # add the bias term
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)


#display the data points:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
