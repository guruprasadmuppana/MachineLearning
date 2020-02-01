# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:33:31 2020

@author: Guru Prasad Muppana
Visualization of MNIST images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mnist():
    path = "D:\\python\\Guru\\machinelearning\\machine_learning_examples-master\\Large_Files\\train.csv"
    
    df = pd.read_csv(path) # reads into a data frame.
    data = df.values # contain X and Y values. first values is its value and 
    # the remianing 28*28 columns of data is image.
    # shuffle the values before extracting X and Y
    np.random.shuffle(data) # inpalce shuffling is done.
    Y = data[:,0]  # extracting all rows but with the first columm values    
#    X = data[:,1:] / 255.0 # exracting all rows but all columns from 1 to the remaining columns

    X = data[:,1:] # not scaling.

    return X, Y


def main():
    print("main function starts here")
    X,Y = get_mnist()
    print("Loading finished")


    numbers = len(set(Y))
    for digit in range(numbers):
        Xi = X[Y==digit]
        Xi = Xi[:784]
        plt.imshow(Xi,cmap="gray")
        title = "Digit - {0}".format(digit)
        plt.title(title)
        plt.show()
         
    
    
    
if __name__ == "__main__" :
    main();