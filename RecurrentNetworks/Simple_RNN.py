# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:36:20 2020

@author: Guru Prasad Muppana

Recurrent Networks 

ANN with time components (Implemented as loop back with shared weights)

NOTE: Though the file name says that it is simple RNN. It is not RNN . It is simple ANN class
with multiple hidden layers. 

We have done mupltiple combinations for 4-bit sequence. It finally worked 
with some combinations below:
    learning_rate=1e-4,
    print_period=8, 
    epochs = 50000,
    batch_sz = 8
    period is 1000
    
    model = ANN([2**(4-1)])
    
Data set is XOR problem which cannot be resolved by simple Linear equations.
    

"""

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle


def all_parity_seq(bit):
    # Generate a table to generate a table with Ntotal binary numbers:
    # however, we are making Ntotal which is a kind of multiple of 100.
    # 
    N = 2**bit # N is the number of possible binary sequences with 0s and 1s
    reminder = 10 - N % 10 # finding out the reminder 
    Ntotal = N + reminder # NTotal is a number with multiple of 100.
    #print(Ntotal)
    X = np.zeros((Ntotal,bit)) # it contians the real binary values for a given a digit wiht bit 0s and 1s
    Y = np.zeros(Ntotal) # Initialize X and Y with zeros first
 
    for i in range(Ntotal):
        a = i % N # repeating after 2**bits sequence until the number is rounded until 100 multiples.
        for j in range(bit): # for every bit of the number.
            if (a % 2**(j+1)) != 0:
                X[i,j] = 1
                a -= 2**(j)
                #print (a)
            
        sum_of_ones = X[i].sum()
        Y[i] = sum_of_ones % 2
    
    return X,Y # X containts the binary digit but the bit order from right to left instead of right to left

def parity_seq(bit):
    # Generate a table to generate a table with Ntotal binary numbers:
    # however, we are making Ntotal which is a kind of multiple of 100.
    # 
    N = 2**bit # N is the number of possible binary sequences with 0s and 1s
    #print(Ntotal)
    X = np.zeros((N,bit)) # it contians the real binary values for a given a digit wiht bit 0s and 1s
    Y = np.zeros(N) # Initialize X and Y with zeros first
 
    for i in range(N):
        a = i
        for j in range(bit): # for every bit of the number.
            if (a % 2**(j+1)) != 0:
                X[i,j] = 1
                a -= 2**(j)
                #print (a)
            
        sum_of_ones = X[i].sum()
        Y[i] = sum_of_ones % 2
    
    return X,Y # X containts the binary digit but the bit order from right to left instead of right to left


    
def init_wieght(Mi,Mo):
    return np.random.randn(Mi,Mo)/np.sqrt(Mi+Mo)

def test1():
    X, Y = all_parity_seq(3)
    print(X)
    print(Y)

def test2():
    print(init_wieght(3,3))


# This class takes X and output is Y with one hidden layer.
# However, the hidden layer has two input : X, itself.

class HiddenLayer:
    def __init__(self,Mi,Mo,layer_id):
        self.Mi = Mi
        self.Mo = Mo
        self.id = layer_id
        
        W = init_wieght(Mi,Mo)
        b = np.zeros(Mo)
        
        self.W = theano.shared(W,"W_%s" % self.id)
        self.b = theano.shared(b,"b_%s" % self.id)
        self.params = [self.W, self.b]
        
    def forward(self,X):
        return T.nnet.relu(X.dot(self.W) + self.b)
        

class ANN(object): # why do we need object
    def __init__(self,hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
    
    def fit(self, X,Y, learning_rate = 1e-2, mu=0.99, reg=1e-12, epochs = 400, batch_sz = 8, print_period=1, show_fig=False):
        
               # X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        
        # setup network. X(t) -> Sequence of hidden layers - > y(t)
        self.hidden_layers = [] # it contains the dimenetional data with wieght information will be there.
        # Set of input layer i.e X
        N, D = X.shape # N is the number of samples :  D is the bit size or features.
        K  = len(set(Y)) # it contains 0s and 1s only. so, size will be 2.
        Mi = D # contains the dimention info for input layer.
        
        layer_id = 0
        print("hidden layer sizes:",self.hidden_layer_sizes)
        
        for layer in self.hidden_layer_sizes :  # note that it is list of hidden layers with sizes
            h = HiddenLayer(Mi,layer, layer_id)
            self.hidden_layers.append(h)
            Mi = layer
            layer_id +=1
        W = init_wieght(Mi,K)
        b = np.zeros(K)
        
        self.W = theano.shared(W,"W_logreg")
        self.b = theano.shared(b,"b_logreg")
        
        self.params = [self.W,self.b]
        for h in self.hidden_layers:
            self.params += h.params # Creating a chain of Ws and B for both initial x(t) and serious of h(t)
        print(self.params)
        
        # for momentums
        dparams = [ theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        
       # cache = [ theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
    
            # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        n_batches = N // batch_sz
        costs = []
        error = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                c, p = train_op(Xbatch, Ybatch)

                if i % 1000 == 0:
#                if j % print_period == 0:
                    costs.append(c)
                    e = np.mean(Ybatch != p)
                    error.append(e)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
            plt.plot(error)
            plt.show()

    

        
    def forward(self,X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)  # uses relu functions for hidden layer.
        
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY,axis=1)



def wide():
    X,Y = parity_seq(4)
    #model = ANN([2048]) for 12
    #model.fit(X,Y, learning_rate=1e-4,print_period=100, epochs = 300, show_fig=True)

    model = ANN([2**(4-1)])
    model.fit(X,Y, learning_rate=1e-4,print_period=8, epochs = 50000, show_fig=True)

def deep():

    X,Y = parity_seq(12)
    model = ANN([1024]*2)
    model.fit(X,Y, learning_rate=1e-3,print_period=100, epochs = 100, show_fig=True)


if __name__ == "__main__":
    print("Main function starts here")
    
    #wide()
    #deep()
    
    
    X, Y = parity_seq(4)
    print(X)
    print(Y)
#    model = ANN([2**(4-1)])
    model = ANN([2**(4)])
    model.fit(X,Y, learning_rate=1e-4,print_period=8, epochs = 50000, show_fig=True)


    #wide()
    # 2**(4-1) = 2**3 = 8
#    model = ANN([8]*2)
#    model.fit(X,Y, learning_rate=1e-4,print_period=1000, epochs = 40000, show_fig=True)
   
#    test1()
#    test2()
    
#    X, Y = all_parity_seq(3)
##    print(X)
##    print(Y)
#    model = ANN([10])
#    model.fit(X,Y)
