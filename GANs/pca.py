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

from sklearn.decomposition import PCA


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
    pca = PCA()
    reduced = pca.fit_transform(X)
    print("Reduced Dimension {0} in seconds {1} ".format( reduced.shape, timelapse.datetime.now() - t0))
    
    #display the first two principla components in scatter plot
    x = reduced[:,0]  # rows are not shulffed but the columns are shuffled.
    y = reduced[:,1]
    z = reduced[:,2]
    
    plt.scatter(x,y,s=10,c=Y, alpha= 0.5)
    plt.title("first and second columns/components")
    plt.show()

    for i in range(10):
        data = reduced[Y==i]
        color = Y[Y==i]
        x = data[:,0]  # rows are not shulffed but the columns are shuffled.
        y = data[:,1]
        plt.scatter(x,y,s=10,c=color, alpha= 0.5)
        plt.title("Digit ")
        plt.show()


    
#    plt.scatter(x,z,s=10,c=Y, alpha= 0.5)
#    plt.title("first and third columns/components")
#    plt.show()
    
    
    plt.plot(pca.explained_variance_ratio_)
    plt.show()
    
    
    for i in range(10):
        data = reduced[Y==i]
        X1 = data
        X1_mean = np.mean(X1,axis=0)
        print(X1_mean.shape)
        image = plt.imshow(X1_mean.reshape((28,28)), cmap="gray")
        plt.show(image)
    
    


if __name__ == "__main__" :
    main()
