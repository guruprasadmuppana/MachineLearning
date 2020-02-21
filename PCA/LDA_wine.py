# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 06:20:25 2020

@author: Guru Prasad Muppana 
LDA (linear Discriminent Analayis)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# load the data from wive dataset
# identify the classification into independant and dependant variable
df = pd.read_csv("wine.csv")
#print(df.columns)
# First column is category.
Y = df.Wine
#print(len(Y))
X = df.drop(['Wine'],axis=1)
#len(X)

# split data into train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=0)

# are we not supposed to shuffle since the data is ordered base on Wine Column


# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest) # are we using the same parameters to transfor Test data.
# is it possible that the test can have its own parameters.

lda = LDA(n_components=2)  # finally, we want to project it on two components.
xtrain = lda.fit_transform(xtrain,ytrain)  # why classifications
xtest = lda.transform(xtest)
#print(lda.get_params())

# xtrain and xtest data is tranformed into different axis with reduced dimentions
#print (xtrain.shape, xtest.shape)

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(multi_class='auto',solver='lbfgs') # multi_classto avoid warnings.
model_lr.fit(xtrain,ytrain )

# find out the score
#print(model_lr.score(xtest,ytest))

# predict value
py = model_lr.predict(xtest)

# display the confusion matric instead of score

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,py) # compare the actual valeus with predicted values
print(cm)

# visualzation of classification using the reduced dimention data.
# we can diplay both train and test sets

from matplotlib.colors import ListedColormap
colormap = ListedColormap(("red","blue","green"))

#https://www.geeksforgeeks.org/numpy-meshgrid-function/
x0_min = xtrain[:,0].min()
x0_max = xtrain[:,0].max()
x1_min = xtrain[:,1].min()
x1_max = xtrain[:,1].max()
y_set= np.unique(ytrain)
step = 0.01
sep = 1

#x1_s = np.linspace(-4, 4, 9) 
#x2_s = np.linspace(-5, 5, 11) 
xx0,xx1 = np.meshgrid(
        np.arange(x0_min - sep, x0_max + sep, step),
        np.arange(x1_min - sep, x1_max + sep, step)
        ) 
#print(xx0)
#print(xx1)

predicted_grid = model_lr.predict(np.array([xx0.ravel(),xx1.ravel()]).T).reshape(xx0.shape)


plt.contourf(xx0,xx1,predicted_grid, alpha = 0.25,cmap=colormap)
plt.xlim(xx0.min(),xx0.max())
plt.ylim(xx1.min(),xx1.max())

for i, j in enumerate(y_set) : # three categories only.
    x = xtrain[ytrain==j,0]
    y = xtrain[ytrain==j,1]
    plt.scatter(x,y, c=colormap(i), label=j)

plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.title("Logistic Regression clasification on DLA axis")
plt.legend()
plt.colorbar()
plt.show() 

