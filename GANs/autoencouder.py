# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 06:30:50 2020

@author: Guru Prasad Muppana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from sklearn.utils import shuffle

def get_mnist():
    path = "D:\\python\\Guru\\machinelearning\\machine_learning_examples-master\\Large_Files\\train.csv"
    
    df = pd.read_csv(path) # reads into a data frame.
    data = df.values # contain X and Y values. first values is its value and 
    # the remianing 28*28 columns of data is image.
    # shuffle the values before extracting X and Y
    #np.random.shuffle(data) # inpalce shuffling is done.
    Y = data[:,0]  # extracting all rows but with the first columm values    
    X = data[:,1:] / 255.0 # exracting all rows but all columns from 1 to the remaining columns
    X, Y = shuffle(X, Y)
    return X, Y


#  df = pd.read_csv('../large_files/train.csv')
#  data = df.values
#  # np.random.shuffle(data)
#  X = data[:, 1:] / 255.0 # data is from 0..255
#  Y = data[:, 0]
#  X, Y = shuffle(X, Y)
#  if limit is not None:
#    X, Y = X[:limit], Y[:limit]
#  return X, Y

class Autoencoder:
    def __init__(self,D,M):
        print("init")
        # store the input samples/data
        self.X = T.matrix('X')
        
        #set up autoencoder:
        # first layer : encoder
        self.W = theano.shared(np.random.randn(D,M)*np.sqrt(2.0/M))
        self.b = theano.shared(np.zeros(M))
        
        #second layer : Deconder
        self.V = theano.shared(np.random.randn(M,D)*np.sqrt(2.0/D))
        self.c = theano.shared(np.zeros(D))
        
        # forward passes.
        self.Z = T.nnet.relu(self.X.dot(self.W) + self.b)
        self.X_hat = T.nnet.softmax(self.Z.dot(self.V)+self.c)
        
        #cost function
        self.cost = T.sum(
                T.nnet.binary_crossentropy(
                        output=self.X_hat,
                        target=self.X
                )
        )
                
        # Gradient decent i,e. optimizing the costs.
        # w <- w - learningrate*grade(cost with respect wights)
        # compact for:
        # local variables ?
        params  = [self.W, self.b, self.V, self.c]
        grads = T.grad(self.cost,params)
        
        #rmsprop: (variable learning rates)
        decay = 0.9 
        lr = 0.001 
        
        # initialize the cache with 1s similar to params shape. It is an array of caches 
        # for each wights and baises.
        cache = [theano.shared(np.ones_like(p.get_value())) for p in params]
        
        # can we remove p in the equation below as it is not used ?
        new_cache = [decay*c + (1-decay)*g*g for p,c,g in zip(params, cache,grads)]
        
        # update cache and then params (Ws and bs)
        updates = [
              (c,new_c) for c, new_c in zip(cache, new_cache)  
        ] + [
            (p, p-lr*g/T.sqrt(new_c + 1e-10))for p,new_c, g in zip(params, new_cache,grads)      
        ]
        
        # define the functions:
        # train and predict
        #train
        
        self.train = theano.function(
                inputs = [self.X],
                outputs = self.cost,
                updates = updates
                )

        # predict function. Basially, it will forward pass 
        self.predict = theano.function(
                inputs = [self.X], # inputs is an array ? output ?
                outputs = self.X_hat
                )


    def fit(self, X, epochs=30,batch_sz=64):
        print("fit")
        costs = []
    
        #N,_ = X.shape()
        N = len(X)
        batches = N//batch_sz
        
        for i in range(epochs):
            print("iteration: ",i)
            # shuffle data.
            np.random.shuffle(X)
            # create batches
            for j in range(batches):
                Xj = X[j*batch_sz:(j+1)*batch_sz]
                c = self.train(Xj)
                c /= batch_sz
                costs.append(c)
                if j % 100 ==0 :
                    print("iteration %d and batch:%d, cost: %0.5f" % (i,j,c))
            plt.plot(costs)
            plt.show()
    

def main():
    print("main function starts here")
    
    X, Y = get_mnist()
    print("loading finished")
    model = Autoencoder(784,300)
    model.fit(X)
    
    
     # plot reconstruction
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = model.predict([x]).reshape(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(im, cmap='gray')
        plt.title("Reconstruction")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
          done = True
        
    
if __name__ == "__main__":
    main()


