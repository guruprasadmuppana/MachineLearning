# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:20:16 2020
@author: Guru Prasad Muppana

Regression using the previous d values.

Data file can be downloaded from :
  https://archive.ics.uci.edu/ml/machine-learning-databases/00291/
  
  The file contains no header. The last column is the target value and all other features are 
  independant variables.
  
  We will use SKlearn Linear Regression to train this supervisor algo.
  RamdomForestRegression from ensemble module
  MLP (Multi layer Perceptron from NeuralNetworks

 # Note that linear Regression is not giving predictions. it is about 50%. 
 # not sure how to improve by using LR hyper parameters.

"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read the file and check the content.
df = pd.read_csv("airfoil_self_noise.dat",sep='\t',header=None)

# first five columns are independant variable
# teh last column is dependant column.

#X = df[:,:-1]  numpy way. It will not work. We need to use explicit columns
#Y = df[:,1]

X = df[[0,1,2,3,4]].values # this includes for all rows.
Y = df[5].values

xtrain,xtest,ytrain, ytest = train_test_split(X,Y,test_size=0.1)
#print(xtrain.shape)
#print(ytrain.shape)
#print(xtest.shape)
#print(ytest.shape)

model = LinearRegression()
model.fit(xtrain,ytrain)

print("train score:",model.score(xtrain,ytrain))
print("test score:",model.score(xtest,ytest))

#predictions = model.predict(xtest)

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor()
model_rf.fit(xtrain,ytrain)

print("train score:",model_rf.score(xtrain,ytrain))
print("test score:",model_rf.score(xtest,ytest))


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# first scale the data
scalar_x = StandardScaler()
xtrain_mlp = scalar_x.fit_transform(xtrain) # it calculates the mean and std from the data and stories as part of fit function 
xtest_mlp = scalar_x.transform(xtest) # using the calculated mean and std, the data is transformed.

scalar_y = StandardScaler()
# Note that y is a vactor. Using exanddim, we will make appropiate required dimensions
# ravel flattens data array
ytrain_mlp = scalar_y.fit_transform(np.expand_dims(ytrain,-1)).ravel()
ytest_mlp = scalar_y.fit_transform(np.expand_dims(ytest,-1)).ravel()


mlp_model = MLPRegressor(max_iter=1000)

mlp_model.fit(xtrain_mlp,ytrain_mlp)


print("train score:",mlp_model.score(xtrain_mlp,ytrain_mlp))
print("test score:",mlp_model.score(xtest_mlp,ytest_mlp))




