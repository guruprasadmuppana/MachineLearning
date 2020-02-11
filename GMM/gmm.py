# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:26:34 2020

@author: Guru Prasad Muppana

GMM : Gausiyan Mixture models.

GMMs :
    This is unsupervised algo for create a class. This is similar to K-NN model, however, 
    the selection of class is done through probalities instead of hard classification. 
    This is also called soft classification since allocate the sample points to categroies with probabalities
    Initially it will start with random guassians.
    There are three parameters for each class:
        pi, mu, and co-variance 
    pi is the probabilit of selecting a give kth class. it is propotional to teh size of the class.
    mu and co-variance are the estimated mean and estimated co-variance of the guasssianss.
    
    For this exerices: 
        We will create a sample two-D points with three guassians. We will plot and see
        And we will pass the same data points to GMM with parameter K
        Note that we will check the process of selecting K in this exercise.

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def clouds(show_fig=True):
    # Generate sample data points with three clusters.
    D = 2 # the samples are two -dimenational. For easy visualization. We can be make it even 1
    N = 2000 # total number of samples.
    sep = 10 # separation between the means
    
    mu1 = np.array([0,0])
    mu2 = np.array([sep,0])
    mu3 = np.array([0,sep])
    
    sigma1 = 2
    sigma2 = 1
    sigma3 = 0.5
    
    X = np.zeros((N,D)) # contains two -data points
    # First cluster contains : 1200, second 600 and third one contains 200
    X[:1200,:] = np.random.randn(1200,D)*sigma1 + mu1
    X[1200:1800,:] = np.random.randn(600,D)*sigma2 + mu2
    X[1800:,:] = np.random.randn(200,D)*sigma3 + mu3
    if show_fig :
        plt.scatter(X[:,0],X[:,1],label="Orignal")
        plt.title("Original Data points")
        plt.legend()
        plt.show()
    return X

class GMM:
    def __init__(self,K):
        self.K = K
        self.max_iteration = 100
        self.smoothing = 1e-2
        
    def displayParameter(self):
        print("Means: \n",self.Means)
        print("Covariances:\n",self.co_vars)
        print("Cluster propoationates or probabilites:\n",self.pis)
        print("Responsbilities size\n", self.R.shape)


    def fit(self, X):
        N, D = X.shape
        print ("We are in fit function")
        
        # Let initialize GMM  parameters.
        self.Means = np.zeros((self.K,D))
        self.co_vars = np.zeros((self.K, D, D))
        self.pis = np.ones(self.K)/self.K # uniform distribution.
        
        # Assignments or Responsbility R holds the samples with probabilities.
        self.R = np.zeros((N,self.K))
        
        #self.displayParameter()
    

        for k in range(self.K):
            self.Means[k] = X[np.random.choice(N)] # randomly select k points and assigning them as means of the cluster
            self.co_vars[k] = np.eye(D, D) # assuming that the gausians are spirical with unit size

        
        lls = []
        weighted_pdfs = np.zeros((N, self.K)) # Collect the probabilities of each sample with k guaisians
        for i in range(self.max_iteration):
        
            # E-Step : calcualte the probabilities of each sample with the given gausians.
            # assign them to R.
            for k in range(self.K):
                weighted_pdfs[:,k] = self.pis[k]*multivariate_normal.pdf(X, self.Means[k], self.co_vars[k])
            # R is the NxK array. For each sample, we have k guasian. 
            #We are keeping converting the probabilies into row.sum = 1 vertors
            self.R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
    
    
            # M step. Recalcualte the M, Co-vars and Pis
            for k in range(self.K): # for each categories, calculate Mu, Sigma/Co-variance and pi.
                # Calculating the pi values.i.e size of the cluster.
                Nk = self.R[:,k].sum() # sums all the wigthed probabilities of for a given category. 
                self.pis[k] = Nk/N
#                print("new size",self.pis[k])

                # Calculate Means:
                self.Means[k] = self.R[:,k].dot(X) / Nk  # not divide by N
                
                #Calcualte co-variances:
                # (X - mu)**2
                
                delta = X - self.Means[k] # N x D
                Rdelta = np.expand_dims(self.R[:,k], -1) * delta # multiplies R[:,k] by each col. of delta - N x D
                self.co_vars[k] = Rdelta.T.dot(delta) / Nk + np.eye(D)*self.smoothing # D x D



            # find the difference error.
            ll = np.log(weighted_pdfs.sum(axis=1)).sum()
            lls.append(ll)
            if i > 0:
                if np.abs(lls[i] - lls[i-1]) < 0.01:
                    print ("iterations:",i)
                    break



                
            print("LL:",lls)
            plt.plot(lls)
            plt.title("Log-Likelihood")
            plt.show()
        
            random_colors = np.random.random((self.K, 3))
            colors = self.R.dot(random_colors)
            plt.scatter(X[:,0], X[:,1], c=colors)
            plt.show()
        
            print("pi:", self.pis)
            print("means:", self.Means)
            print("covariances:", self.co_vars)
            return self.R





if __name__ == "__main__" :
    print("GMM functionality starts here")

    # display the cloud of data points    
    X = clouds()

    #  Cluster classification using GMM
    # Assume that there are three clusters.
    k = 3
    g = GMM(k)
    g.fit(X)
