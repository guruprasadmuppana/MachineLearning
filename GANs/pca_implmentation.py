# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:16:28 2020

@author: Guru Prasad Muppana

Principle Component analysis

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as timelapse

#from sklearn.decomposition import PCA


def get_mnist():
    path = "D:\\python\\Guru\\machinelearning\\machine_learning_examples-master\\Large_Files\\train.csv"
    
    df = pd.read_csv(path) # reads into a data frame.
    data = df.values # contain X and Y values. first values is its value and 
    # the remianing 28*28 columns of data is image.
    # shuffle the values before extracting X and Y
    np.random.shuffle(data) # inpalce shuffling is done.
    Y = data[:,0]  # extracting all rows but with the first columm values    
    X = data[:,1:]  # exracting all rows but all columns from 1 to the remaining columns
    return X, Y






def main():
    print("main starts here")
    t0 = timelapse.datetime.now() 
    X, Y = get_mnist()
    print("data load finished in ", timelapse.datetime.now() - t0 )
    
    X1 = X [Y== 1] # collecting only digit 1 and finding out displaying the image
    X1_mean = np.mean(X1,axis=0)
    print(X1_mean.shape)
    image = plt.imshow(X1_mean.reshape((28,28)), cmap="gray")
    plt.show(image)
    
   
    
    t0 = timelapse.datetime.now() 
    
    # find co-variance of X (data NxD)
    covX = np.cov(X.T)
    print("Covariance of X",covX.shape)
    # find eigen vectors and eigen values:
    lambdas,Q = np.linalg.eigh(covX)
    print("Lambda size",lambdas.shape) # it should be a vector of D
    print("Q size",Q.shape) # it should be DxD

    plt.plot(lambdas)
    plt.show()
        
    
    # sorted the lemdas from  from smallest to largest and find its index
    # some may be negative..
    idx = np.argsort(-lambdas) # making from largest to smallest
    #get sorted lembdas
    lambdas = lambdas[idx] # order the columns.
    # if the lambda value is zero, reset to zero.
    #lambdas = np.maximum(lambdas,0)
    print(np.min(lambdas))

    
    #Similarly order columns of Q (eigen vectors)
    Q = Q[:,idx] # DxD matric : columns are inter changed
    
    Z = X.dot(Q) # Coverance of Z and X will be the same. Z is diagonal matric 
    #and its diagonal values are coverance of each column
    print("Z size ", Z.shape)
    
    plt.scatter(Z[:,0],Z[:,1],s=10,c=Y,alpha = 0.3)
    plt.title("first and second columns of Z values")
    plt.show()
    
    
    for i in range(10):
        data = Z[Y==i]
        color = Y[Y==i]
        x = data[:,0]  # rows are not shulffed but the columns are shuffled.
        y = data[:,1]
        plt.scatter(x,y,s=10,c=color, alpha= 0.5)
        plt.title("Digit ")
        plt.show()


    
    plt.plot(lambdas)
    plt.show()
    
    
    for i in range(10):
        data = Z[Y==i]
        X1 = data
        X1_mean = np.mean(X1,axis=0)
        print(X1_mean.shape)
        image = plt.imshow(X1_mean.reshape((28,28)), cmap="gray")
        plt.show(image)
    

#    for i in range(10):
#        data = Z[Y==i]
#        X1 = data[:700] #
#        X1 = np.mean(X1,axis=0)
#        print(X1.shape)
#        image = plt.imshow(X1.reshape(28,28), cmap="gray")
#        plt.show(image)
#    


if __name__ == "__main__" :
    main()
