# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:39:33 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal as mvn


def get_mnist():
    path = "D:\\python\\Guru\\machinelearning\\machine_learning_examples-master\\Large_Files\\train.csv"
    
    df = pd.read_csv(path) # reads into a data frame.
    data = df.values # contain X and Y values. first values is its value and 
    # the remianing 28*28 columns of data is image.
    # shuffle the values before extracting X and Y
    np.random.shuffle(data) # inpalce shuffling is done.
    Y = data[:,0]  # extracting all rows but with the first columm values    
    X = data[:,1:] / 255.0 # exracting all rows but all columns from 1 to the remaining columns
    return X, Y

class BayesClassifier:
    # K -> number of classification. In this case, it runs from 0-9. K=10
    # we will calculate the mean and coverinace for each set K and story them in a dict object
    # calcualte the probability of identify an image (sample) for each category using counting technique.
    def fit(self,X,Y):
        self.K = len(set(Y)) # provides unique values of Y array.
        self.p_y = np.zeros(self.K) # to store probabilities of each category k
        self.gaussians = [] # the gaussians distribution model parameters are mean and coverance/standard deviation
        N = len(X) # total sample size. i.e. total number of images.
        for i in range(self.K):
            # collect all images whose category is k.
            Xi = X[Y==i]
            Xi_mean= np.mean(Xi,axis=0)
            Xi_cov = np.cov(Xi.T) # transpose of Xi
            g = {"mean":Xi_mean,"cov":Xi_cov}
            self.gaussians.append(g)
            self.p_y[i] = float(len(Xi)/N) # converting to float to avoid any loss of data/information.
        
    def clamp(self,a):
        a = np.minimum(a,1)
        a = np.maximum(a,0)
        return a
    
    def sample_given_y(self,y):
         g = self.gaussians[y]
         return self.clamp(mvn.rvs(mean=g["mean"],cov=g["cov"])) # makes it black and while
         #return mvn.rvs(mean=g["mean"],cov=g["cov"])
    
    
    def sample(self):
        y = np.random.choice(self.K,self.p_y)
        return self.sample_given_y(y)
    
def main():
    print("main function starts here")
    #1. Get mnist data
    X,Y = get_mnist()
    bayes_classifier = BayesClassifier()
    bayes_classifier.fit(X,Y)
#    
#    #display mean images
#    for i in range(10):
#        g = bayes_classifier.gaussians[i]
#        mean = g["mean"]
#        cov = g["cov"]
#        plt.imshow(mean.reshape(28,28),cmap="gray")
#        plt.show()
    
    for k in range(bayes_classifier.K):
        sample = bayes_classifier.sample_given_y(k).reshape(28,28)
        mean = bayes_classifier.gaussians[k]["mean"].reshape(28,28)
         
        plt.subplot(1,2,1)
        plt.imshow(sample,cmap="gray")
        plt.title("Sample")
         
        plt.subplot(1,2,2)
        plt.imshow(mean,cmap="gray")
        plt.title("Mean")
         
        plt.show()
         

    
if __name__ == "__main__" :
    main()
