# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:19:13 2020

@author: Guru Prasad Muppana

This is simple RNN with one hidden layer which is looping back to itself.


"""
import numpy as np
import os
from nltk import pos_tag, word_tokenize
#from datetime import datetime
from sklearn.utils import shuffle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]



def get_poetry_classifier_data(samples_per_class, load_cached=True, save_cached=True):
    datafile = 'poetry_classifier_data2.npz'
    load_cached = False
    if load_cached and os.path.exists(datafile):
        npz = np.load(datafile)
        X = npz['arr_0']
        Y = npz['arr_1']
        V = int(npz['arr_2'])
        return X, Y, V

    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('edgar_allan_poe.txt', 'robert_frost.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print(line)
                # tokens = remove_punctuation(line.lower()).split()
                tokens = get_tags(line)
                if len(tokens) > 1:
                    # scan doesn't work nice here, technically could fix...
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print(count)
                    # quit early because the tokenizer is very slow
                    if count >= samples_per_class:
                        break
    if save_cached:
        np.savez(datafile, X, Y, current_idx)
    return X, Y, current_idx


class SimpleRNN:
    def __init__(self,M,V):
        self.M = M # Number of states.
        self.V = V # vacobular size (Disctionary size)
        
        
        
    def set (self,Wx,Wh,bh,h0,Wo,bo,activation):
        self.f = activation
        
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        
        self.params = [self.Wx, self.Wh, self.bh,self.h0, self.Wo, self.bo]
        
        thX = T.ivector('X')
        thY = T.iscalar('Y')
        
        def recurrence(x_t,h_t1): # inner function.
            # returns h(t), y(t)
            h_t  = self.f(self.Wx[x_t]+ h_t1.dot(self.Wh)+self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t,y_t
        
        [h,y], _ = theano.scan(
                fn = recurrence,
                outputs_info = [self.h0, None],
                sequences = thX,
                n_steps = thX.shape[0], # first parameter of shape i,e. N in N*D
        )
        
        py_x = y[-1,0,:]  # what is this?
        
        prediction = T.argmax(py_x)
        self.predict_op = theano.function(
                inputs=[thX],
                outputs = prediction,
                allow_input_downcast = True,
        )
        return thX,thY,py_x,prediction


    
    def fit(self,X, Y, learning_rate = 1.0, mu= 0.99, reg=1.0, activation=T.tanh, epochs = 500, show_fig=False ) :
        M = self.M
        V = self.V
        
        K = len(set(Y)) # number of classficiations. In this case, two poets type.
        
        print("Vocabulary size", V)
        
        # Shuffle data ... before we train:
        X,Y = shuffle(X,Y)
        
        # split the data into train and test.
        
        Nvalid = 10 # for testing.
        Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:]
        X, Y = X[:-Nvalid],Y[:-Nvalid]
        
        N = len(X)  # number of samples i.e. sequences.
        
        # Setup RNN
        Wx = init_weight(V,M)
        Wh = init_weight(M,M)
        bh = np.zeros(M)
        h0 = np.zeros(M) # This is the initial hidden layer values. Will be zero
        
        Wo = init_weight(M,K)
        bo = np.zeros(K)
        
        thX, thY, py_x, prediction = self.set(Wx,Wh,bh,h0,Wo,bo,activation)
        
        # cost calculations.
        
        cost = - T.mean(T.log(py_x[thY]))
        grads = T.grad(cost,self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        lr = T.scalar("learning_rate")
        
        updates = [
                (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
                (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]
        
        self.train_op = theano.function(
            inputs = [thX, thY,lr],
            outputs = [cost, prediction],
            updates = updates,
            allow_input_downcast = True,
        )
        
        # trainiing
        costs = []
        for i in range(epochs):
            print("iteration",i)
            X, Y = shuffle(X,Y)
            n_correct = 0
            cost = 0
            for j in range(N): # for all N samples.
                #print("X[%d]",j)
                c, p = self.train_op(X[j],Y[j],learning_rate)
                cost += c
                if p == Y[j] :
                    n_correct +=1
                    
            learning_rate *= 0.9999 # decreasing learning rate
            
            # calcualte validation accuracy
            n_correct_valid = 0
            for j in range(Nvalid):
                p = self.predict_op(Xvalid[j])
                if p == Yvalid[j]:
                    n_correct_valid +=1
#            print("i:",i, "cost:",cost,"correct rate",(float(n_correct)/N),end=" ")
            print("validation correct rate", (float(n_correct_valid)/Nvalid))
            costs.append(cost)
            
        if show_fig:
            plt.plot(costs)
            plt.show()
    
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        Wx = npz['arr_0']
        Wh = npz['arr_1']
        bh = npz['arr_2']
        h0 = npz['arr_3']
        Wo = npz['arr_4']
        bo = npz['arr_5']
        V, M = Wx.shape
        rnn = SimpleRNN(M, V)
        rnn.set(Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn
    


if __name__ == "__main__":

    X, Y, V = get_poetry_classifier_data(samples_per_class=40)
    print("Hello")
    rnn = SimpleRNN(30, V)
    rnn.fit(X, Y, learning_rate=1e-6, show_fig=True, activation=T.nnet.relu, epochs=50)

