# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:05:12 2020

@author: Guru Prasad Muppana

Bagging concepts applied to classification.

We will use random numbers spread across a squere with four different regions


"""

# imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

############# UTIL

# plot decision boundary.
def plot_counter(X, model):
    # calcualate boundaries meshgrid.
    h = 0.02
    x_min, x_max = X[:,0].min() - 1,X[:,0].max() + 1 # one unit extended at the both sides
    y_min, y_max = X[:,1].min() - 1,X[:,1].max() + 1 # one unit extended at the both sides
    # an array of points between x_min # x_max with step size h
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h), 
                        np.arange(y_min,y_max,h) 
                        )
    
#    print("Count:",np.arange(x_min,x_max,h).shape)
#    print("xx:",xx.shape)
#    print("yy:",yy.shape)
    # Plot the decision boundary.
    # We will assign a color to each point in the mesh size : [x_min,x_max]*[y_min,y_max] 
    # ravel is makes the array into flattened array ; 
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
#    print("Z.shape 1",Z.shape)
    
    # put the result into color map.
    Z = Z.reshape(xx.shape)
#    print("Z.shape 1",Z.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired) # counterf() can be tried.

################## UTIL


np.random.seed(10) # check with multiple values including "not defining case"

# creating a data set. it will be 500 points divided into four regions.
N = 500
# binary classification :
D = 2 # it is 2 dimentional data (x,y). like a set of points. 
# but each point can be either belong class 1 or class 2.
X = np.random.randn(N,D)

# Two guasians
sep = 1.5  # try different values like 1 and 3. 3 give 100% score since the points are well seperated.
X[:N//2] += np.array([sep,sep])
X[N//2:] += np.array([-sep,-sep])
Y = np.array([0]*(N//2) + [1]*(N//2))
X,Y = shuffle(X,Y)

## XOR model
#sep = 2.0  # try 3, 4 .... etc. it will increase the percentage is since separation is more.
## Following are hard-coded but it can be generic as well.
#X[:125] += np.array([sep,sep]) # move random ponits to (sep,sep)
#X[125:250] += np.array([sep,-sep])
#X[250:375] += np.array([-sep,-sep])
#X[375:] += np.array([-sep,sep]) 
#Y = np.array([0]*125 + [1]*125 + [0]*125 +[1]*125)

plt.scatter(X[:,0],X[:,1],s=50,alpha=0.5,c=Y)
plt.show()

# Single DT classification with Depth = None (no limts)
model = DecisionTreeClassifier(max_depth=None)
model.fit(X,Y)
print("Score with 1 D.Tree",model.score(X,Y))

plot_counter(X,model)
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.5,c=Y)
plt.show()

class BaggedTreeClassifier:
    def __init__(self,B):
        self.B = B
        
    def fit(self,X,Y):
        # create B bags.
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N,size=N,replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth=2) # Limited depth. week calssifier.
            model.fit(Xb,Yb)
            self.models.append(model)
            
    def predict(self, X):
        
        N = len(X)
        predictions = np.zeros(N) # initialize with zero since we are going to element wise operation
        for model in self.models:
            prediction = model.predict(X)
            predictions += prediction
        #print("last prediction",prediction)
        return np.round(predictions/self.B) # element wise division.
        # Note : predictions are added across all bagging sampling.
        # for a given point, if more than 50% bagging sampels gives voting as 0 or 1, we select that as its class
        # by since we are doing element wise addition B time, we need to divide the same with B
        
        
    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(Y==P)
    




# Bagged Tree Classifier.
#model = DecisionTreeClassifier(max_depth=None)
model = BaggedTreeClassifier(200) # 200 B bagging samples
# If the number of B is less, score drops heavily.
model.fit(X,Y)
print("Score with Bagged Tree",model.score(X,Y))
# BaggedTree gives in the range of 96.8% for seed(10) case. However, some times, it gives in the range of 80s

plot_counter(X,model)
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.5,c=Y)
plt.show()










