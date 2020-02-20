# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:10:32 2020

@author: Guru Prasad Muppana 

We will use airline passenger information to predict the future values.

We will use Linear Regression, Nueral Networks and RNN 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Modules from Sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from datetime import datetime


# Read the file

df = pd.read_csv("international-airline-passengers.csv", skipfooter=3, engine='python')
# change the column names.
df.columns = ["month","passengercount"]
#print(df.columns)

# split the data into train and test.
X = df["month"].values  # converts into integer. this is not used at all.
print("total size:",len(X))
Y = df["passengercount"].values
#visual data
#plt.plot(Y)
#plt.show()


# one dimentional data
series = df["passengercount"].values



# We will use T past values to decide the next value.
# we will have to create a new test data :
# in this new data, we will have T-sample as input and dependant varaiable as Y
#T = 5
train_acc = []
test_acc= []

#for T in (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40):

# Learning. In general,  more the past values , better is the predictions. However, based on the data pattern,
# Past data should be at least one cycle of pattern.

for T in ([50]):

    n = len(X)-T # numebr of sampels with T
    X_seq = np.zeros((n,T))
    Y_seq = np.zeros(n)
    
    for t in range(len(X)-T):
        # the new data set will be n*T
        X_seq[t] = series[t:t+T]
        Y_seq[t] = series[t+T]
    
    xtrain, xtest , ytrain, ytest = train_test_split(X_seq,Y_seq, test_size=0.3)
    #print(xtrain.shape, xtest.shape , ytrain.shape, ytest.shape)
    
    model = LinearRegression()
    model.fit(xtrain,ytrain)
    print("Train score:",model.score(xtrain,ytrain))
    print("Test score:",model.score(xtest,ytest))
    train_acc.append(model.score(xtrain,ytrain))
    test_acc.append(model.score(xtest,ytest))
    
    py= model.predict(X_seq)
    # display the orginal and predicted value:
    
    # add T seq of number nan values.
#    plt.plot(Y,label="original")
#    plt.plot(np.concatenate([np.full(T, np.nan), py]),label="Predicted value")
#    plt.title("Prediction using Linear Regression from SKlearn")
#    plt.legend()
#    plt.show()

plt.plot(train_acc,label="train")
plt.plot(test_acc,label="train")
plt.legend()
plt.show()


#print(X_seq[0])
#print(Y_seq[0])
#print(X_seq.shape)
#print(Y_seq.shape)

##########################################################
# ANN

import theano
import theano.tensor as T
from sklearn.utils import shuffle

def init_weight(M1, M2):
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)

def myr2(T, Y):
    Ym = T.mean()
    sse = (T - Y).dot(T - Y)
    sst = (T - Ym).dot(T - Ym)
    return 1 - sse / sst

class HiddenLayer(object):
    def __init__(self, M1, M2, f, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return self.f(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, activation=T.tanh, learning_rate=1e-3, mu=0.5, reg=0, epochs=5000, batch_sz=None, print_period=100, show_fig=True):
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        # initialize hidden layers
        N, D = X.shape
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W = np.random.randn(M1) / np.sqrt(M1)
        b = 0.0
        self.W = theano.shared(W, 'W_last')
        self.b = theano.shared(b, 'b_last')

        if batch_sz is None:
            batch_sz = N

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.vector('Y')
        Yhat = self.forward(thX)

        rcost = reg*T.mean([(p*p).sum() for p in self.params])
        cost = T.mean((thY - Yhat).dot(thY - Yhat)) + rcost
        prediction = self.forward(thX)
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

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
        )

        n_batches = np.int(N / batch_sz)
        # print "N:", N, "batch_sz:", batch_sz
        # print "n_batches:", n_batches
        costs = []
        for i in np.arange(epochs):
            X, Y = shuffle(X, Y)
            for j in np.arange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if (j+1) % print_period == 0:
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c)
        
#        if show_fig:
#            plt.plot(costs)
#            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return Z.dot(self.W) + self.b

    def score(self, X, Y):
        Yhat = self.predict_op(X)
        return myr2(Y, Yhat)

    def predict(self, X):
        return self.predict_op(X)


################ train #################

series = series.astype(np.float32)
series = series - series.min()
series = series / series.max()
N = len(series)
# 2 previouse values:    

train_acc=[]
test_acc=[]
    
#[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
for seq_size in ([20]):

    n = len(X)-seq_size # numebr of sampels with T
    X_seq = np.zeros((n,seq_size))
    Y_seq = np.zeros(n)
    
    for t in range(len(X)-seq_size):
        # the new data set will be n*T
        X_seq[t] = series[t:t+seq_size]
        Y_seq[t] = series[t+seq_size]
    
    xtrain, xtest , ytrain, ytest = train_test_split(X_seq,Y_seq, test_size=0.3)
    #print(xtrain.shape, xtest.shape , ytrain.shape, ytest.shape)
    
    model = ANN([100]) # 200 is the original value
    model.fit(xtrain, ytrain, activation=T.tanh)
    
    print("ANN Train score:",model.score(xtrain,ytrain))
    print("ANN Test score:",model.score(xtest,ytest))
    train_acc.append(model.score(xtrain,ytrain))
    test_acc.append(model.score(xtest,ytest))
    
    py= model.predict(X_seq)
    # display the orginal and predicted value:
    
    # add T seq of number nan values.
    plt.plot(series,label="original")
    plt.plot(np.concatenate([np.full(seq_size, np.nan), py]),label="Predicted value")
    plt.title("Prediction using Linear Regression from ANN")
    plt.legend()
    plt.show()

plt.plot(train_acc,label="train")
plt.plot(test_acc,label="train")
plt.title("ANN")

plt.legend()
plt.show()


#from rnn_class.lstm import LSTM
#from rnn_class.gru import GRU

class GRU:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxr = init_weight(Mi, Mo)
        Whr = init_weight(Mo, Mo)
        br  = np.zeros(Mo)
        Wxz = init_weight(Mi, Mo)
        Whz = init_weight(Mo, Mo)
        bz  = np.zeros(Mo)
        Wxh = init_weight(Mi, Mo)
        Whh = init_weight(Mo, Mo)
        bh  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br  = theano.shared(br)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz  = theano.shared(bz)
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh  = theano.shared(bh)
        self.h0  = theano.shared(h0)
        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.h0]

    def recurrence(self, x_t, h_t1):
        r = T.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        z = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        hhat = self.f(x_t.dot(self.Wxh) + (r * h_t1).dot(self.Whh) + self.bh)
        h = (1 - z) * h_t1 + z * hhat
        return h

    def output(self, x):
        # input X should be a matrix (2-D)
        # rows index time
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0],
            n_steps=x.shape[0],
        )
        return h




#def init_weight(M1, M2):
#    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)

#def myr2(T, Y):
#    Ym = T.mean()
#    sse = (T - Y).dot(T - Y)
#    sst = (T - Ym).dot(T - Ym)
#    return 1 - sse / sst



class RNN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
    # epochs=2000 
    def fit(self, X, Y, activation=T.tanh, learning_rate=1e-1, mu=0.5, reg=0, epochs=100, show_fig=False):
        N, t, D = X.shape

        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = GRU(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = np.random.randn(Mi) / np.sqrt(Mi)
        bo = 0.0
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        lr = T.scalar('lr')
        thX = T.matrix('X')
        thY = T.scalar('Y')
        Yhat = self.forward(thX)[-1]

        # let's return py_x too so we can draw a sample instead
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=Yhat,
            allow_input_downcast=True,
        )
        
        cost = T.mean((thY - Yhat)*(thY - Yhat))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.train_op = theano.function(
            inputs=[lr, thX, thY],
            outputs=cost,
            updates=updates
        )

        costs = []
        for i in np.arange(epochs):
            t0 = datetime.now()
            X, Y = shuffle(X, Y)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in np.arange(N):
                
                c = self.train_op(learning_rate, X[j], Y[j])
                cost += c
            if i % 10 == 0:
                print ("i:", i, "cost:", cost, "time for epoch:", (datetime.now() - t0))
            if (i+1) % 500 == 0:
                learning_rate /= 10
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.output(Z)
        return Z.dot(self.Wo) + self.bo

    def score(self, X, Y):
        Yhat = self.predict(X)
        return myr2(Y, Yhat)

    def predict(self, X):
        N = len(X)
        Yhat = np.empty(N)
        for i in np.arange(N):
            Yhat[i] = self.predict_op(X[i])
        return Yhat


# standardized will be used. 
#series = series.astype(np.float32)
#series = series - series.min()
#series = series / series.max()
#N = len(series)
## 2 previouse values:    

train_acc=[]
test_acc=[]

for seq_size in ([20]):

    n = len(X)-seq_size # numebr of sampels with T
    X_seq = np.zeros((n,seq_size))
    Y_seq = np.zeros(n)
    
    for t in range(len(X)-seq_size):
        # the new data set will be n*T
        X_seq[t] = series[t:t+seq_size]
        Y_seq[t] = series[t+seq_size]
    
    xtrain, xtest , ytrain, ytest = train_test_split(X_seq,Y_seq, test_size=0.3)
    #print(xtrain.shape, xtest.shape , ytrain.shape, ytest.shape)
    
    
    Ntrain = len(xtrain)
    xtrain = xtrain.reshape(Ntrain, seq_size, 1)
    Ntest = len(xtest)
    xtest = xtest.reshape(Ntest, seq_size, 1)

    
    model = RNN([50])
    model.fit(xtrain, ytrain, activation=T.tanh)

    
    print("RNN Train score:",model.score(xtrain,ytrain))
    print("RNN Test score:",model.score(xtest,ytest))
    train_acc.append(model.score(xtrain,ytrain))
    test_acc.append(model.score(xtest,ytest))
 
    
#    Nseq = len(X_seq)
#    X_seq = X_seq.reshape(Nseq, seq_size, 1)    
#    py= model.predict(X_seq)
    # display the orginal and predicted value:
    
    # add T seq of number nan values.
    plt.plot(series,label="original")
    plt.plot(np.concatenate([np.full(seq_size, np.nan), py]),label="Predicted value")
    plt.title("Prediction using Linear Regression from ANN")
    plt.legend()
    plt.show()

plt.plot(train_acc,label="train")
plt.plot(test_acc,label="train")
plt.title("ANN")

plt.legend()
plt.show()

