# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    # Download this file from Kaggle
    #https://www.kaggle.com/c/digit-recognizer
    train = pd.read_csv('mnist_train.csv').values.astype(np.float32)
    train = shuffle(train)

    split_amount = 10000
    Xtrain = train[:-split_amount,1:] / 255
    Ytrain = train[:-split_amount,0].astype(np.int32)

    Xtest  = train[-split_amount:,1:] / 255
    Ytest  = train[-split_amount:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

# get the data
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# decompose covariance
covX = np.cov(Xtrain.T)
lambdas, Q = np.linalg.eigh(covX)


# lambdas are sorted from smallest --> largest
# some may be slightly negative due to precision
idx = np.argsort(-lambdas)
lambdas = lambdas[idx] # sort in proper order
lambdas = np.maximum(lambdas, 0) # get rid of negatives
Q = Q[:,idx]


# plot the first 2 columns of Z
Z = Xtrain.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s=100, c=Ytrain, alpha=0.3)
plt.show()


# plot variances
plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

# cumulative variance
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()