# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:09:30 2020

@author: Guru Prasad Muppana

using head to brain data, we use sklearn package to solve the same problem.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the file headbrain.cvs file.
# The file is available https://www.kaggle.com/jemishdonda/headbrain 
filename = "headbrain.csv"


data = pd.read_csv(filename)

# Finding the relation ship between head size and brain waights
X = data["Head Size(cm^3)"]  # independant value
Y = data["Brain Weight(grams)"] # Dependant value

# Visualize the data and see if it has any pattern.
plt.scatter(X,Y)
plt.title("Head size Vs Brain wieght")
plt.show()

# Using standard ML librares . SKLearn
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error

size = len(X)
# SKLearn cannot handle series objects. Hence we need to provide 1-dimention data explicity
# Converting panda objects into np arrays.
X_np = np.array(X)
Y_np = np.array(Y)

X_np = X_np.reshape((size,1))

#Create a model
reg = LinearRegression()
# Train the model
reg = reg.fit(X_np,Y_np)
# Reg stores both m and c values internally

# Predict 
Y_predict = reg.predict(X_np)

#plt.plot(X_np,Y_predict,color="red", label="Regressed line from X and Y data")
plt.plot([min(X_np), max(X_np)], [min(Y_predict), max(Y_predict)], color='red',label="Regressed line from X and Y data")  # regression line
plt.scatter(X,Y,c="blue", label="Scatter plot with the original values")
plt.legend()


plt.title("Head size Vs Brain wieght")
plt.xlabel("Head size")
plt.ylabel("brain weight")

plt.show()


R2_score = reg.score(X_np,Y_np) # score internally calcualtes the predicted values . 

print("Model Parameter:")
print("coef", reg.coef_)
print("intercept", reg.intercept_)

print(R2_score)


