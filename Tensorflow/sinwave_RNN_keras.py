# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 07:49:17 2020

@author: Guru Prasad Muppana

We will generate simple sine wave data 
Using Keras RNN model, we will start predict the next output based on given T sequence of previous data.


"""

import numpy as  np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,SimpleRNN, Dense
from keras.optimizers import SGD, Adam


# generate the sinwave and display it. 
# note the sin wave will have some noise .
data_size = 200 # number of points of the sin wave
smoothing = 0.01 # try 0.1  to introduce more noise.

data = np.sin(0.1*np.arange(data_size))+ np.random.randn(data_size)*smoothing
#data = np.arange(data_size) for sechechig the sequence generation

#plt.plot(data)
#plt.show()

# generate T points as a sequence .
T = 10 # the esquence size; the last one is the last predicted value
D = 1 # 
X = [] # will contain the sequences with T size. It is a collection of sequences.
Y = [] # will contian the T+1 predicted values.
for i in range(len(data)-T): # -1 is removed form to include the point
    seq = data[i:i+T]
#    print("x",seq[-1]); note that i+T does not include the final value while selecting elements form array
#    print("seq",seq) # print entire sequence
    X.append(seq)
    next_data_point = data[i+T]
#    print("y",next_data_point)
    Y.append(next_data_point)

# convert python lists into np arrays
X = np.array(X)
Y = np.array(Y)
N = len (X)

X = np.expand_dims(X,-1) # check why this is needed. This is needed while passing input layer.
# there we need T,D
# it is moved 190x10 -> 190x10x1

# split the train and test sets
x_train, y_train = X[:-N//2], Y[:-N//2]
x_test, y_test = X[N//2:],Y[N//2:]

i = Input(shape=(T,D))
h1 = SimpleRNN(5)(i)
o = Dense(1)(h1) # note that we are predicte regression value. No softmax is needed

# Build the model
model = Model(i,o)

# set up cost and optiomizaiton function:

model.compile(loss="mse", optimizer=Adam(lr=0.1)) # mse since it is a loss function.

# train the model
r = model.fit(x_train,y_train,
              validation_data=(x_test,y_test),
              epochs=100) # Try 10 and 100. See the difference between 10 and 100. Accuracy increases when tried mored.

# plot the losses
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

outputs = model.predict(X)
print(outputs.shape)
predictions= outputs[:,0] # output is 190x1. Picking up first column

#predictions= outputs # output is 190x1. Picking up first column. Even this will work


plt.plot(Y, label="targets")
plt.plot(predictions,label="predictions")
plt.title("Simple RNN")
plt.legend()
plt.show()

