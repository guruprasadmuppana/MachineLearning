# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 09:54:18 2020

@author: Guru Prasad Muppana

Researching how KNN and DT (Decision Trees behave in boosting techniques with respect to Bias Vs Variance trade off)

"""

#imports
import numpy as np
import matplotlib.pyplot as plt

# import SK learn libraries 
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.utils import shuffle

# Create a data set for varification. 
# Real function is Sin(3X)
# Approximator is DT and KNN. We will start with DT

N = 20  # points
Ntrain = 12 # training set points

X = np.linspace (0, 2*np.pi,N).reshape(N,1) # By default, linspace gives a vector. 
# we are ensuring that it is Nx1 array.
Y = np.sin(3*X) # No noise since it is a true function.
X,Y = shuffle(X,Y) # why is this requried?S

#Collect training set:
Xtrain = X[:Ntrain]
Ytrain = Y[:Ntrain]


# display the data set.
# c = color ; s=size of the point,
plt.scatter(Xtrain, Ytrain, c="blue", s=50,alpha= 0.4)

# Let us apprioximate using DT
model = DecisionTreeRegressor(max_depth=None)
model.fit(Xtrain,Ytrain)

plt.scatter(Xtrain,model.predict(Xtrain).reshape(Ntrain,1), c="green", s=50, alpha=0.6)
plt.title("Decision Tree - Low bias , high variance")
# Using different data set but using the above model
T=50
Xaxis = np.linspace(0,2*np.pi,T)
Yaxis = np.sin(3*Xaxis) # not used .

plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis,model.predict(Xaxis.reshape(T,1)))

# Note: We are displaying the original function in line mode. Blue points are derived from 
# the same original function. Green dots are predicted points using DT. 
# you notice that these points are on the same line. i.e low bias.
# you also notice that the original points and predicted points are very close.
# hence the low bias.
plt.show()

# The same is tested using max depth is limited to 1.

model = DecisionTreeRegressor(max_depth=1)
model.fit(Xtrain,Ytrain)

plt.scatter(Xtrain, Ytrain, c="blue", s=50,alpha= 0.4)
plt.scatter(Xtrain,model.predict(Xtrain).reshape(Ntrain,1), c="green", s=50, alpha=0.6)
plt.title("Decision Tree - high bias , low variance")
# Using different data set but using the above model
T=50
Xaxis = np.linspace(0,2*np.pi,T)
Yaxis = np.sin(3*Xaxis) # not used .

plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis,model.predict(Xaxis.reshape(T,1)))

# Note: We are displaying the original function in line mode. Blue points are derived from 
# the same original function. Green dots are predicted points using DT with max-depth = 1. 
# you notice that green dots are away from blue points i.e high bias.
# you also notice that the predicted points are not on the line.
# hence the low bias.
plt.show()

# using KNN

# Case: high low bias and high  variance
model = KNeighborsRegressor(n_neighbors=1)
model.fit(Xtrain,Ytrain)

plt.scatter(Xtrain, Ytrain, c="blue", s=50,alpha= 0.4)
plt.scatter(Xtrain,model.predict(Xtrain).reshape(Ntrain,1), c="green", s=50, alpha=0.6)
plt.title("KNN - Low bias , high variance")
# Using different data set but using the above model
T=50
Xaxis = np.linspace(0,2*np.pi,T)
Yaxis = np.sin(3*Xaxis) # not used .

plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis,model.predict(Xaxis.reshape(T,1)))

# Note: We are displaying the original function in line mode. Blue points are derived from 
# the same original function. Green dots are predicted points using DT. 
# you notice that these points are on the same line. i.e low bias.
# you also notice that the original points and predicted points are very close.
# hence the low bias.
plt.show()

# The same is tested using n_neighbors = 10 

model = KNeighborsRegressor(n_neighbors=10)
model.fit(Xtrain,Ytrain)

plt.scatter(Xtrain, Ytrain, c="blue", s=50,alpha= 0.4)
plt.scatter(Xtrain,model.predict(Xtrain).reshape(Ntrain,1), c="green", s=50, alpha=0.6)
plt.title("KNN - high bias , low variance")
# Using different data set but using the above model
T=50
Xaxis = np.linspace(0,2*np.pi,T)
Yaxis = np.sin(3*Xaxis) # not used .

plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis,model.predict(Xaxis.reshape(T,1)))

# Note: We are displaying the original function in line mode. Blue points are derived from 
# the same original function. Green dots are predicted points using DT with max-depth = 1. 
# you notice that green dots are away from blue points i.e high bias.
# you also notice that the predicted points are not on the line.
# hence the low bias.
plt.show()


# let us do the same thing for classification.
# let us get some data points:

N=100
D=2

X = np.random.randn(N,D)
# divide the points into (1,1) and (-1,-1) as their means.
X[:N//2] = X[:N//2] + np.array([1,1]) # Center at (1,1)
X[N//2:] = X[N//2:] + np.array([-1,-1]) # Center at (-1,-1)
Y = np.array(     [0]*(N//2) + [1]*(N//2)   ) # first half is marked as zero and the sceond half is marked with 1s

plt.scatter(X[:,0],X[:,1],s=50,c=Y, alpha= 0.5) 
plt.title("Original data")
plt.show()

# Use decision classifier to separate with depth = None (unlimited depth)

# plot decision boundary.
def plot_decision_boundary(X, model):
    # calcualate boundaries meshgrid.
    h = 0.02
    x_min, x_max = X[:,0].min() - 1,X[:,0].max() + 1 # one unit extended at the both sides
    y_min, y_max = X[:,1].min() - 1,X[:,1].max() + 1 # one unit extended at the both sides
    # an array of points between x_min # x_max with step size h
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h), 
                        np.arange(y_min,y_max,h) 
                        )
    
#    print("Count:",np.arange(x_min,x_max,h).shape)
#    print("xx:",xx.shape)
#    print("yy:",yy.shape)
    # Plot the decision boundary.
    # We will assign a color to each point in the mesh size : [x_min,x_max]*[y_min,y_max] 
    # ravel is makes the array into flattened array ; 
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
#    print("Z.shape 1",Z.shape)
    
    # put the result into color map.
    Z = Z.reshape(xx.shape)
#    print("Z.shape 1",Z.shape)
    plt.contour(xx,yy,Z,cmap=plt.cm.Paired) # counterf() can be tried.



model = DecisionTreeClassifier(max_depth=None)
model.fit(X,Y) # training.

plt.scatter(X[:,0],X[:,1],s=50,c=Y, alpha= 0.5) # plots the original data
plot_decision_boundary(X, model)
plt.title("Decision Classifer : low biase and high variance")
plt.show()

model = DecisionTreeClassifier(max_depth=2) # two classifications
model.fit(X,Y) # training.

plt.scatter(X[:,0],X[:,1],s=50,c=Y, alpha= 0.5) # plots the original data
plot_decision_boundary(X, model)
plt.title("Decision Classifer : high biase and low variance")
plt.show()

# KNN

model = KNeighborsClassifier(n_neighbors=1) # each points as classifere
model.fit(X,Y) # training.

plt.scatter(X[:,0],X[:,1],s=50,c=Y, alpha= 0.5) # plots the original data
plot_decision_boundary(X, model)
plt.title("KNN : low biase and high variance")
plt.show()

model = KNeighborsClassifier(n_neighbors=20) # each points as classifere
model.fit(X,Y) # training.


plt.scatter(X[:,0],X[:,1],s=50,c=Y, alpha= 0.5) # plots the original data
plot_decision_boundary(X, model)
plt.title("KNN : high biase and low variance")
plt.show()


# Meshgrid exmpale:
x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

xx,yy = np.meshgrid(x,y,sparse=True) # False will create 100x100 array for xx, yy
# else xx.shape will be (1,100) or yy.shape will be (100,1) 
print (xx.shape)
print (yy.shape)


z = np.sin(xx**2+yy**2)/(xx**2+yy**2)
h = plt.contourf(x,y,z) # 'f' stands for filled countours
#h = plt.contour(x,y,z) # 'f' stands for filled countours
plt.show()

















