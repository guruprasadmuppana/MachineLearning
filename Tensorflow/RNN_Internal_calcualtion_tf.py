# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:08:40 2020

@author: Guru Prasad Muppana

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D) # a random sample

# Build simple RNN
M = 5 # number of hidden units
i = Input(shape=(T, D)) # 10x3 input size. 10 samples with 3 dimention data points.
x = SimpleRNN(M)(i)
x = Dense(K)(x) # out is two but it does not have any sigmoid. So, it is simple regression


model = Model(i, x)


# Predict the sample based on the current weights. 
# note that the current weights are randomly generated.
Yhat = model.predict(X)
print(Yhat)  # note done value


# See the how the config of Simple RNN
# Get the weights first
model.summary()


# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_2 (InputLayer)         [(None, 10, 3)]           0         
# _________________________________________________________________
# simple_rnn_1 (SimpleRNN)     (None, 5)                 45        
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 12        
# =================================================================
# Total params: 57
# Trainable params: 57
# Non-trainable params: 0

# Get the first layer : hidden layer which connect from input and itself.
model.layers[1].get_weights()

# Check their shapes
# Should make sense
# First output is input > hidden
# Second output is hidden > hidden
# Third output is bias term (vector of length M)
a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

# (3, 5) (5, 5) (5,) -> Cross check with the above details.

# collect the weights from the layers and manual calculate the first pass.
Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()


h_last = np.zeros(M) # initial hidden state
x = X[0] # the one and only sample
Yhats = [] # where we store the outputs


for t in range(T):
  h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
  y = h.dot(Wo) + bo # we only care about this value on the last iteration
  Yhats.append(y)
  
  # important: assign h to h_last
  h_last = h


# print the final output
print(Yhats[-1])


