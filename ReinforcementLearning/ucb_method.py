# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:29:16 2020

@author: UCB method to findout the true "mean"
"""
import numpy as np
import matplotlib.pyplot as plt

from comparing_epsilon import run_experiment as run_experiment_eps

class Bandit:
    def __init__(self,m):
        self.m = m
        self.N = 0
        self.mean = 0
        
    def pull(self):
        # return value from machine with m. 
        return self.m + np.random.randn() # random sample from normal distribution with mean value a m.

    # update the mean value (running mean)
    def update(self,x): 
        self.N +=1
        self.mean = (1-1.0/self.N)*self.mean + (1.0/self.N)*x
# Bandit class ends here.

def ucb(mean, n, nj):
    # return value with uppper confidence bound
    # mean + sqrt(2*ln(n)/jth count of Bandit)
    if nj == 0:
        return float('inf')
    return mean + np.sqrt(2*np.log(n)/nj)



           
def run_experitment(m1,m2,m3,N):
    
    bandits = [Bandit(m1),Bandit(m2),Bandit(m3)]
    data = np.empty(N)
    
#    for i in range(N) : # for all N samples
#        # Greedy algo: Select maximim value:
#        j = np.argmax([ucb(b.mean,i+1,b.N)] for b in bandits)
#        bandit = bandits[j] # select the best bandi with max mean value.
#        x = bandit.pull() # select a sample from jth Bandit (best Bandit)
#        bandit.update(x) # udpate jth Bandit.
#        # for plot
#        data[i] = x

    for i in range(N):
       j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])
       x = bandits[j].pull()
       bandits[j].update(x)
    
       # for the plot
       data[i] = x



    cumulative_average = np.cumsum(data)/(np.arange(N)+1)
    
    # plot
    plt.plot(cumulative_average);
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale("log")
    plt.show()
    
    
    return cumulative_average
    
# run_experiment ends here.
    

if __name__ == "__main__" :
    
    ucb = run_experitment(1.0,2.0,3.0,10000)
    eps = run_experiment_eps(1.0,2.0,3.0,0.05,10000)
 
      # log scale plot
    plt.plot(eps, label='eps = 0.05')
    plt.plot(ucb, label='ucb1')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    
      # linear plot
    plt.plot(eps, label='eps = 0.05')
    plt.plot(ucb, label='ucb1')
    plt.legend()
    plt.show()


    
    
