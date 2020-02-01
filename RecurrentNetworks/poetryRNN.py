# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:23:04 2020

@author: Guru Prasad Muppana 

Poetry generation

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import theano
import theano.tensor as T

from wikipedia import get_robert_frost

def init_weight(Mi,Mo):
    return np.random.randn(Mi,Mo)/np.sqrt(Mi+Mo)


class PoetryRNN:
    def __init__(self, D,M,V):
        # D : 
        self.D = D  # size of word embebeding size.
        self.M = M # hidden layer size
        self.V = V # vacabulory size
        
    def fit(self, X, learning_rate=1., mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V
        self.f = activation

        # initial weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        # make them theano shared
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix
        thY = T.ivector('Y')

        # sentence input:
        # [START, w1, w2, ..., wn]
        # sentence target:
        # [w1,    w2, w3, ..., END]

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu*dp - learning_rate*g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_total = sum((len(sentence)+1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                # problem! many words --> END token are overrepresented
                # result: generated lines will be very short
                # we will try to fix in a later iteration
                # BAD! magic numbers 0 and 1...
                input_sequence = [0] + X[j]
                output_sequence = X[j] + [1]

                # we set 0 to start and 1 to end
                c, p = self.train_op(input_sequence, output_sequence)
                # print "p:", p
                cost += c
                # print "j:", j, "c:", c/len(X[j]+1)
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])
        

def train_poetry():
    sentences, word2idx = get_robert_frost()
#    rnn = SimpleRNN(30, 30, len(word2idx))
#    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
#    rnn.save('RNN_D30_M30_epochs2000_relu.npz')

if __name__ == "__main__":
    sentences, word2idx = get_robert_frost()
    print(len(word2idx))
    rnn = PoetryRNN(30,30,len(word2idx))
    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('RNN_D30_M30_epochs2000_relu.npz')


