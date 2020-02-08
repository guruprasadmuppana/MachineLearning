# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:57:46 2020

@author: Guru Prasad Muppana

perceptron: early stage neural network. These perceptrons do not have any hidden layers.

Assumptions:
    only three input parameters for input layer.
    Output layer has only one value



"""
import numpy as np

from matplotlib import pyplot as plt

#np.random.seed(1)



# input and out data
# it has four samples.
X = np.array([[0,0,1],
             [1,1,1],
             [1,0,1],
             [1,1,0],
             [1,0,0],             
             [0,0,0],             
             [0,1,1]])
Y = np.array([[0,1,1,1,1,0,0]]).T # Row is converted into column.


# util functions
def sigmod(x):
    return 1/(1+np.exp(-x))
def sigmod_der(x):
    tempx = sigmod(x)
    #tempx = x
    return tempx*(1-tempx)

class Perceptron:
    def __init__(self):
        # w = sinapses's weights. this includes the bais. w0 can be used for bais
        self.w = 2*np.random.random((3,1)) -1  # ramdom valeus with mean value at zero range is [-1,1]
        self.epchos = 100
        
    def forward (self, X,W):
        # X is 4x3 matric; W is 3X1 matric (vector)
        return sigmod(np.dot(X,W))
        
    def fit(self,X,Y):
        #print(self.w)    
        
        for iteration in range(self.epchos):
            # calculate the out of perceptron using the current wieghts.
            output = self.forward(X,self.w)
            
            # error:
            #error = output - Y
            error =  Y - output

            # adjust the wieghts based on error and gradients
            # adj = dot (input , error*der(output))
            
            output_derivate = sigmod_der(output)
            change_in_error = error*output_derivate
            
            adjustment = np.dot(X.T,change_in_error)
            self.w += adjustment
        
        
        pass;
    def predict(self,X):
        return self.forward(X,self.w)
        pass
    def score(self,y,y_hat):
        #return np.mean(y == y_hat)
        return np.mean(y == np.round(y_hat))



if __name__ == "__main__":
    print("main function starts here")
    perceptron = Perceptron()
    print("Inital weights:\n",perceptron.w)
    perceptron.fit(X,Y)
    print("Final weights:\n",perceptron.w)
    p_y = perceptron.predict(X)
    print(perceptron.score(Y,p_y))
    

#    Check with other type of data
    
