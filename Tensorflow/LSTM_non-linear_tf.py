# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:36:16 2020

@author: Guru Prasad Muppana

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


from tensorflow.keras.layers import Input,  Dense, Flatten
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM


# input data:
# it is single dimention data. 

series = np.sin((0.1*np.arange(400))**2)

plt.plot(series)
plt.show()

# let's see if we can use T past values to predict the next value
T = 10
D = 1  # It is single dimention data.
X = []  # it is a collection of T sequeence of past T values.
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T) # make it N x T
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

#X.shape (390, 10) Y.shape (390,)

# # model 1 :
# # Auto Regressive (ANN model)

# ### try autoregressive linear model
# i = Input(shape=(T,))  # No RNN or LSTM or GRU.
# x = Dense(1)(i)
# model = Model(i, x)
# model.compile(
#   loss='mse',
#   optimizer=Adam(lr=0.01),
# )

# # X[:-N//2] first half of the samples.

# # train:
# r = model.fit(
#   X[:-N//2], Y[:-N//2],
#   epochs=80,
#   validation_data=(X[-N//2:], Y[-N//2:]),
# )

# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

# # losses falls to ground over training session.
# # NOTE:
# # using past T values (T=10), we can predict the values of non-linear equation
# # like sin with t**2 (non-linear with the time)


# # one step forecast using true targets
# # Note: even the one-step forecast fails badly
# outputs = model.predict(X)  # X includes all points. Each point is Tx1 sequences. NxTx1
# print(outputs.shape)
# predictions = outputs[:,0]

# plt.plot(Y, label='targets')
# plt.plot(predictions, label='predictions')
# plt.title("Linear Regression Predictions")
# plt.legend()
# plt.show()

# # for non-linear equations, AR model does not work though 
# # it is trying to simulate similar values


# # Trying to predict the second half values of the curve.
# validation_target = Y[-N//2:]
# validation_predictions = []

# # index of first validation input
# i = -N//2

# while len(validation_predictions) < len(validation_target):
#   p = model.predict(X[i].reshape(1, -1))[0,0] # 1x1 array -> scalar
#   i += 1
  
#   # update the predictions list
#   validation_predictions.append(p)

# plt.plot(validation_target, label='forecast target')
# plt.plot(validation_predictions, label='forecast prediction')
# plt.legend()
# plt.title("LR Predictions using one sample at a time.")
# plt.show()


# # Multi-step forecast
# validation_target = Y[-N//2:]
# validation_predictions = []

# # last train input
# last_x = X[-N//2] # 1-D array of length T

# while len(validation_predictions) < len(validation_target):
#   p = model.predict(last_x.reshape(1, -1))[0,0] # 1x1 array -> scalar
  
#   # update the predictions list
#   validation_predictions.append(p)
  
#   # make the new input
#   last_x = np.roll(last_x, -1)
#   last_x[-1] = p

# plt.plot(validation_target, label='forecast target')
# plt.plot(validation_predictions, label='forecast prediction')
# plt.title("LR Predictions using Multi-step. reusing predicted points")
# plt.legend()
# plt.show()
# # first level predictions are wrong. Now, these are fed back into predictions.
# # interesting, the predictions are becoming zero values.


########### We will use RNN/LSTM for non-linear data points:
### Now try RNN/LSTM model
X = X.reshape(-1, T, 1) # make it N x T x D. It is a requirement for RNN/GRU/LSTM

# make the RNN
i = Input(shape=(T, D)) # D = 1.
x = LSTM(10)(i)  # 10 hidden units in the hidden layer.
#x = SimpleRNN(10)(i)  # 10 hidden units in the hidden layer.
# Try various combination of RNN, GRU and LSTM
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.05),
)

# train the RNN
r = model.fit(
  X[:-N//2], Y[:-N//2],
  batch_size=32,
  epochs=200,
  validation_data=(X[-N//2:], Y[-N//2:]),
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title("many-to-one RNN /LSTM- One step")
plt.legend()
plt.show()


# Multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) < len(Y[-N//2:]):
  # Reshape the input_ to N x T x D
  f = model.predict(input_.reshape(1, T, 1))[0,0]
  forecast.append(f)

  # make a new input with the latest forecast
  input_ = np.roll(input_, -1)
  input_[-1] = f

plt.plot(Y[-N//2:], label='targets')
plt.plot(forecast, label='forecast')
plt.title("RNN/LSTM Forecas - Multi step")
plt.legend()
plt.show()