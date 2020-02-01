# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:22:31 2020

@author: Guru Prasad Muppana

Ensemble Machine Learning:
    Bias vs Variance trade off.

"""

# imports:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# Algo
# Variance is the how much error varias when the trained over the multiple
# set of sampless.
# We take data set X. However, for training purpose, we take NUM_DATASETS from X
# for training using appropimator function.

# For this exerices, our true function is sin(x) and 
# approximator is linear regression with d dimentions.

# We proof that variance increase when the d increase after certain value.
# here d is the degree of the polymonial function (Approximator)

np.random.seed(2) # fixing for every run

# true function:
def f(X):
    return np.sin(X)

# make a dataset with X^D, X^(D-1), X^0
def make_polynominal(x,D):
    # x is set of N samples.
    N = len(x)
    X = np.empty((N,D+1))
    for d in range(D+1): # d is columns
        X[:,d] = x**d
        if d > 1 :
            X[:,d] = (X[:,d] - X[:,d].mean())/X[:,d].std()
    return X

# display the true function:
x_axis = np.linspace(-np.pi,np.pi,100) # drawing sin wave with 100 points.
y_axis = f(x_axis)
#plt.plot(x_axis,y_axis)
#plt.axhline(xmin=-np.pi,xmax=np.pi)
#plt.show()

# Create data set for the above true function with N points
N = 25 # samples points 

X = np.linspace(-np.pi,np.pi,N)
np.random.shuffle(X) # not sure why  if we do not shuffle, the out put  is quite different.
f_X = f(X) # selected N point on X and created N points on Y using f(x)
#plt.plot(X,f_X)
#plt.axhline(xmin=-np.pi,xmax=np.pi)
#plt.show()

# Maximum number of polynomial degree is 12
MAX_POLY = 12

Xpoly = make_polynominal(X,MAX_POLY)

# Number of data sets is 50 
NUM_DATASETS = 30 # from 9 to 14, it is showing of degree is 6 instead of 3.
Ntrain = int(N*0.9)
NOISE_VARIANCE = 0.5


train_scores = np.zeros((NUM_DATASETS,MAX_POLY))
test_scores = np.zeros((NUM_DATASETS,MAX_POLY))


train_predictions = np.zeros((Ntrain,NUM_DATASETS,MAX_POLY))
prediction_curves = np.zeros((100,NUM_DATASETS,MAX_POLY))


# running the model  with multiple training sets.

model = LinearRegression()

for k in range(NUM_DATASETS):
    # create kth train and test data with X and Y
    Xtrain = Xpoly[:Ntrain] # 26 columns ; 90% of the points
    Y = f_X + np.random.randn(N)*NOISE_VARIANCE # error. i.e. simulated data points collected from the field.
    Ytrain = Y[:Ntrain]
#    print("min:",np.min(Y))
#    print("max:",np.max(Y))
    
    
    Xtest = Xpoly[Ntrain:]
    Ytest = Y[Ntrain:]
    
#    plt.plot(Xpoly[:,1],Y)
#    plt.axhline(xmin=-np.pi,xmax=np.pi)
#    plt.show()
    
    for d in range(MAX_POLY):
        model.fit(Xtrain[:,:d+2],Ytrain) # first iteration takes only d=2 complexity equation.
        predictions = model.predict(Xpoly[:,:d+2])
    
        x_axis_poly = make_polynominal(x_axis,d+1)
        prediction_axis = model.predict(x_axis_poly)
        
#        if (d+1 == 12):
#            print("min:",np.min(prediction_axis))
#            print("max:",np.max(prediction_axis))
#        
#        plt.plot(x_axis,prediction_axis,color="green", alpha=0.5)
#        plt.show()
       
        prediction_curves[:,k,d] = prediction_axis
       
        train_prediction = predictions[:Ntrain]
        test_prediction = predictions[Ntrain:]
       
        train_predictions[:,k,d] = train_prediction
        
        train_score = mse(train_prediction,Ytrain)
        test_score = mse(test_prediction,Ytest)
        
        train_scores[k,d] = train_score
        test_scores[k,d] = test_score
       
        
        

for d in range(MAX_POLY):
    for k in range(NUM_DATASETS):
        plt.plot(x_axis, prediction_curves[:,k,d],color="green", alpha=0.5)
    plt.plot(x_axis,prediction_curves[:,:,d].mean(axis=1),color="blue", linewidth=2.0)
    plt.title("All curevers for degree = %d" % (d+1))
    plt.show()


# calculate the squared bias
avg_train_prediction = np.zeros((Ntrain,MAX_POLY))
squared_bias = np.zeros(MAX_POLY)

f_Xtrain = f_X[:Ntrain]

for d in range(MAX_POLY):
    for i in range(Ntrain):
        avg_train_prediction[i,d] = train_predictions[i,:,d].mean()
    squared_bias[d] = ((avg_train_prediction[:,d] - f_Xtrain)**2).mean()
    

#calculate the variance
variances = np.zeros((Ntrain,MAX_POLY))
for d in range(MAX_POLY):
    for i in range(Ntrain):
        delta = train_predictions[i,:,d] - avg_train_prediction[i,d]
        variances[i,d] = delta.dot(delta)/len(delta)
variance = variances.mean(axis=0)        

degrees = np.arange(MAX_POLY)+1
best_degree = np.argmin(test_scores.mean(axis=0))+1

plt.plot(degrees,squared_bias,label="sqrt bias")
plt.plot(degrees,variance,label="variance")
plt.plot(degrees,test_scores.mean(axis=0),label="test scores")
plt.plot(degrees,squared_bias+variance,label="sqr bias + variance")
plt.axvline(x=best_degree,linestyle="--", label="Best Complexity")
plt.legend()
plt.show()


plt.plot(degrees,train_scores.mean(axis=0),label="train scores")
plt.plot(degrees,test_scores.mean(axis=0),label="test scores")
plt.axvline(x=best_degree,linestyle="--", label="Best Complexity")
plt.legend()
plt.show()





#def main():
#    print("Main function starts here")
#
#if __name__ == "__main__" :
#    main()