# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:22:53 2020

@author: Guru Prasad Muppana.

here we will use two sine waves. Try to predict the next value of the combined value.

"""

import numpy as  np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,SimpleRNN, Dense
from keras.optimizers import SGD, Adam


# generate the sinwave and display it. 
# note the sin wave will have some noise .
data_size = 200 # number of points of the sin wave
smoothing = 0 # try 0.1  to introduce more noise.

wave1 = np.sin(0.1*np.arange(data_size))+ np.random.randn(data_size)*smoothing
wave2 = np.sin(0.2*np.arange(data_size))+ np.random.randn(data_size)*smoothing

#data = np.arange(data_size) for sechechig the sequence generation

plt.plot(wave1)
plt.plot(wave2)
plt.plot(wave1*wave2, linewidth=2)

plt.show()

# generate T points as a sequence .
T = 10 # the esquence size; the last one is the last predicted value
D = 2 # The sequence contains ten points. However, each point contians two values
# i.e two dimension data 
X = [] # will contain the sequences with T size. It is a collection of sequences.
Y = [] # will contian the T+1 predicted values.
for t in range(data_size-T): # -1 is removed form to include the point
    seq = wave1[t:t+T],wave2[t:t+T]
#    print("x",seq[-1]); note that i+T does not include the final value while selecting elements form array
#    print("seq",seq) # print entire sequence
    X.append(seq)
    next_data_point = wave1[t+T]*wave2[t+T]
    # Note that we are using + for combined signal
    # + tring loss: 8.7e-5, 8.9e-5
    # try - 3.05e-5, 3.3e-5
    # Try * 0.0076 and validation error = 0.0048
    
#    print("y",next_data_point)
    Y.append(next_data_point)

# convert python lists into np arrays
X = np.array(X)
Y = np.array(Y)
N = len (X)

# the Follow was used for 1 dimensional signal.
#X = np.expand_dims(X,-1) # check why this is needed. This is needed while passing input layer.
## there we need T,D
## it is moved 190x10 -> 190x10x1

# Convert the size appropiately.

print("X.shape:",X.shape)
X = np.transpose(X, (0,2,1))
# X had 190 points ; each one contains 10 (T) sequence of points. 
# however, each point was contained twp points coming from two different waves.
# So, its shape was: 190x2x10
# since the input layer takes in put TxD. We conver the data into appropiate size.
# i.e 190x10x2. Transpose helps the same. (x,y,z) -> 0 is remains the same position
# 3rd dimension (2) will become second dimension (1). Same way, second dimension data is moved to third demision.


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
plt.title("Many (a sequence of T points) to one RNN")
plt.legend()
plt.show()

