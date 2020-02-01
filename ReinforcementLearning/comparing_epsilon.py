# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:49:12 2020

@author: Guru Prasad Muppana

Slot machine learning algo.
There are three slot machines. 

This algo explain the concepts of explore vs exploit.

We give epsilon,a small number to represent the perentage of exploration.

while exploring if we get a probability less than espilon, we will randomly select 
a slot machines from three of the slot machines

"""
import numpy as np
import matplotlib.pyplot as plt

class SlotMachine:
    def __init__(self, m_id):
        self.m_id = m_id # machine id 
        self.runningmean = 0
        self.N = 0 # the number of data points for a given SlotMachine
        
    def pull(self):
        #return a gaussian normal value from the slot machine for this machine.
        # machine id is added to separate the points between the points for a given machines
        return np.random.randn() + self.m_id
    
    def update(self,x): # Update the machines mean value (a kind of score)
        self.N += 1 # increment the counter for slot machine
        self.runningmean = (1-1/self.N)*self.runningmean + (1/self.N)*x
        
        


def run_experiment(machine1, machine2, machine3, eps,N):
    
    # Create slot machines:
    machines = [SlotMachine(machine1),SlotMachine(machine2),SlotMachine(machine3)]
    
    data = np.empty(N) # create N element array adn initilize with empty values. not zero values.
    x_axis = np.empty(N)
    
    for i in range(N): # take one sample at a time and process it.
        # espilon greedy
        p = np.random.random()  # Uniform distribution . Value is between [0 , 1). Similar to probability distribution 
        if p < eps :
            j = np.random.choice(3) # Generate a random numbers from (0,1,2). This can be used as machine id.
        else:
            # get the best slot machine whose mean values is hightest among three slot machines. 
            j = np.argmax([m.runningmean for m in machines]) 
        
        x = machines[j].pull()
        machines[j].update(x)
        
        x_axis[i] = j + 1 # +1 to move the points from zero.
        
        data[i] = x # adding the outcome from the slot machine i
        
        cumulative_average = np.cumsum(data)/ (np.arange(N)+1)
        # (arange(N)+1) : an array of numbers : 1,2,3,4,.... N
        # np.cumsum(data) is partially populated array with N items. the values gets filled from 0,1,2...
        # It is running average
        
        
#    plt.scatter(x_axis,data)    
#    plt.show()
        
        
    # plot the moviing average CTR (Click trough rate)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*machine1)
    plt.plot(np.ones(N)*machine2)
    plt.plot(np.ones(N)*machine3)
    plt.xscale("log") # log scale
    plt.show()

    for machine in machines:
        print("machine id", machine.m_id)
        print("machine Running Mean", machine.runningmean)
        print("Number of trails (x)", machine.N)
        
    return cumulative_average

        
# try with 100 points, 10 points etc
if __name__ == "__main__" :
    ca_10 = run_experiment(1,2,3,0.1,10000) # 10 %
    ca_5 = run_experiment(1,2,3,0.05,10000) # 5%
    ca_1 = run_experiment(1,2,3,0.01,10000) # 1%

    plt.plot(ca_10,label="eps = 10%")
    plt.plot(ca_5,label="eps = 5%")
    plt.plot(ca_1,label="eps = 1%")
    plt.xscale("log") # log scale
    plt.title("Log daigram")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.plot(ca_10,label="eps = 10%")
    plt.plot(ca_5,label="eps = 5%")
    plt.plot(ca_1,label="eps = 1%")
    plt.title("Normal daigram")
    plt.legend()
    plt.grid(True)
    plt.show()


    
