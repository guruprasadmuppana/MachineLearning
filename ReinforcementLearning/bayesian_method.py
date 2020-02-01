# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:52:53 2020

@author: Basian Bandit

# in this case, we generate sample with given predicted mean (running mean)

"""
import numpy as np
import matplotlib.pyplot as plt


class BayesianBandit:
    # initialize with true mean
    # then other running means
    def __init__(self,true_mean):
        
        self.true_mean = true_mean
        
        self.predicted_mean = 0
        #lemda
        self.lambda_ = 1 # lambda is standard key word. hence extra "_"
        self.sum_x = 0 # temp variable to hold the current sum of x values
        self.tau = 1
        
    def pull(self):
        return np.random.randn() + self.true_mean
    
    def sample(self):
        return np.random.randn()/np.sqrt(self.lambda_) + self.predicted_mean
    
    def update(self,x):
        self.lambda_ += self.tau
        self.sum_x +=x
        # predicted mean = tau*(sum of x /lambda)
        self.predicted_mean = self.tau*(self.sum_x/self.lambda_)
# Class BasianBandit ends here        

def run_experiment(m1,m2,m3,N):
    bandits = [BayesianBandit(m1), BayesianBandit(m2),BayesianBandit(m3)]
    
    data = np.empty(N)
    
    for i in range(N):
        # chose random sample from selected 
        j = np.argmax([b.sample() for b in bandits ])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x
        
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()


if __name__ == "__main__" :
    run_experiment(1.0,2.0,3.0,10000)    

