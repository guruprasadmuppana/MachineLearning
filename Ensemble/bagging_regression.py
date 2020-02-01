# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:29:11 2020

@author: Guru Prasad Muppana 
Bagging Regression

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

# Create data using 
T = 100 # large space of sample
x_axis = np.linspace(0,2*np.pi, T)
#y_axis = np.cos(x_axis) + x_axis*2
y_axis = np.sin(x_axis)


# We will now bag the sample of size 30 and trian the model. 
# Generating training data
N=50 # try various values like 10, 20,30, 40, 50
# N value for 1 and 2, score gives negative. It is not predictiing
# but N value greater than 5 leads 30% accuracy i.e great.
idx = np.random.choice(T, size=N, replace=False) # note : False means "without replacement"
Xtrain = x_axis[idx].reshape(N,1)
Ytrain = y_axis[idx]

#Create one model 
model = DecisionTreeRegressor() # None = meaning , any size of depth.
model.fit(Xtrain,Ytrain)
predictions = model.predict(x_axis.reshape(T,1))
print("Score :", model.score(x_axis.reshape(T,1),y_axis))

plt.plot(x_axis,y_axis) # original function : blue color is first color. 
plt.plot(x_axis,predictions) # predicated function using single decision Tree

plt.show()


# Now try bagging.
# We will use Class approach

class BaggedTreeRegressor:
    def __init__(self,B):
        self.B = B
    def fit(self,X,Y):
        N = len(X)
        self.models = []
        
        for b in range(self.B):
            # Select N size sample and create a model using the N size data point
            idx = np.random.choice(N,size=N,replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeRegressor() # == None, no limit in depth
            model.fit(Xb,Yb)
            self.models.append(model) # collect the B models for each bootstrap sample.
            
        
    def predict(self,X):
        # Since many models are there, we need to average.
        predictions = np.zeros(len(X))
        for model in self.models:
            prediction = model.predict(X)
            predictions = predictions + prediction # element by operation. its size is (len(X),)
#        print("predictions",predictions.shape)
        return predictions / self.B  # averaging
        
        
        return X
    
    
    def score(self,X,Y): # R2 => 1- ((Y-Y^))**2/(Y-Y.mean)**2
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return  1 - d1.dot(d1)/d2.dot(d2)
    
model = BaggedTreeRegressor(200) # custom clas
model.fit(Xtrain,Ytrain)
predictions = model.predict(x_axis.reshape(T,1))
print("Score using Bagged Tree :", model.score(x_axis.reshape(T,1),y_axis))

plt.plot(x_axis,y_axis) # original function : blue color is first color. 
plt.plot(x_axis,predictions) # predicated function using single decision Tree

plt.show()

