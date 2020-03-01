# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:23:39 2020

@author: Guru Prasad Muppana

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# We will apply Linear Regression for polinomial expression

# Generate the data for X^2 curve. 
# the equation that we are interested is : a*X^2 + b*X + C
# this can be written as a*X2 + bX1 + C*X0 ; X2 = X^2; X0 = 1
N = 1000
a = [100,1,0.1]

with open('data_poly.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N) 
    x_original = X
    # Returns 100 points from uniform distribution
    X2 = X*X
    Y = a[2]*X2 + a[1]*X + a[0] + np.random.normal(scale=100, size=N) # Guisian noise is added.
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))

# Read the file and prepare the data for processing.
        
# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x]) # add the bias term x0 = 1
    # our model is therefore y_hat = w0 + w1 * x + w2 * x**2
    Y.append(float(y))

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1],Y, s= 3, alpha = 0.5)  # X[:,0] contains all 1. X[:,2] contains X^2 values. 
# we are displaying X values on x-axis
plt.title("Original Polinominal equation")
plt.xlabel("x-value")
plt.ylabel("a2*X^2 + a1*X + a0")
plt.show()

# We will use linear equations using mathematical model.

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

print("Original wieghts (w0,w1,w2):",a,"\n")
print("Calculated wieghts (w0,w1,w2:", w[0],w[1],w[2])

x_p = X[:,1] # X values.
y_p = w[2]*(x_p**2) + w[1]*x_p + w[0]

sorted_original_x = np.sort(x_original)

y_original_without_noise = a[2]*(sorted_original_x**2) + a[1]*sorted_original_x + a[0]


plt.scatter(x_p,y_p, s=4, alpha = 0.5)  
plt.title("Calculaed Polinominal equation")
plt.xlabel("x-value")
plt.ylabel("w2*X^2 + w1*X + w0")
plt.plot(sorted_original_x,y_original_without_noise, c=(0,1,0,0.5))
plt.show()

# determine how good the model is by computing the r-squared
Yhat = X.dot(w)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
print("Accuracy depends upon noise in the original data")




