# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:58:20 2020

@author: Guru Prasad Muppana

Linear Regression using simple analytical way using dervied formule.

m = Sum( (x-x_bar)*(y-y_bar)/(x-x_bar)**2)
c = y_bar - m*x_bar


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the file headbrain.cvs file.
# The file is available https://www.kaggle.com/jemishdonda/headbrain 
filename = "headbrain.csv"


data = pd.read_csv(filename)

# Finding the relation ship between head size and brain waights
X = data["Head Size(cm^3)"]  # independant value
Y = data["Brain Weight(grams)"] # Dependant value

# Visualize the data and see if it has any pattern.
plt.scatter(X,Y)
plt.title("Head size Vs Brain wieght")
plt.show()

# We will use simple  y = mx +c rule find out values.
# Find out slope i.e m.  m = Sigma((x - x.mean)*(y- y.mean))/sigma(x-x.mean)**2
# c = y.mean - m*x_means
x_mean = X.mean()
y_mean = Y.mean()

m1 = (((X-x_mean)*(Y-y_mean)).sum())
m2 = (((X-x_mean)**2).sum())
m = m1/m2
# c value 
c= y_mean - m*x_mean

# Now plot the line using m, c for given set of x and computed y values
# horizontal line axis:
x_max = X.max()
x_min = X.min()

x = np.linspace(x_min,x_max,1000) # x is np array
# note that x is new set x values ... not given set of x values.
y = m*x + c  # y is np array


plt.plot(x,y,color="lime", label="Regressed line from X and Y data")
plt.scatter(X,Y,c="blue", label="Scatter plot with the original values")
plt.legend()


plt.title("Head size Vs Brain wieght")
plt.xlabel("Head size")
plt.ylabel("brain weight")

plt.show()

# Measuing the fitness using R-Squarred model.
# R**2 = 1 - Sigma(preditected y - y_mean)**2 / sigma (y - y_mean)**2
#ss_t is the total sum of squares and ss_r is the total sum of squares 
# of residuals(relate them to the formula).
y_predicted = m*X +c

ss_r = ((y_predicted - Y)**2).sum()
ss_t = ((Y - y_mean)**2).sum()
R_squarred =  1 - (ss_r/ss_t)


print ("Model Parameter")
print ("m,c :", m,c)

print ("R_squarred",R_squarred)

