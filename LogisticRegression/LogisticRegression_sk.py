# -*- coding: utf-8 -*-
"""


@author: Guru Prasad Mupppana

Binary classification.

"""

import numpy as np
import pandas as pd

# Read the file headbrain.cvs file.
filename = "User_Data.csv"

data = pd.read_csv(filename)

# Finding the relation ship between head size and brain waights
#X = data[["Gender","Age","EstimatedSalary"]]  # independant value
#X.loc[X["Gender"]=="Male",["Gender"]]=1
#X.loc[X["Gender"]=="Female",["Gender"]]=0
X = data[["Age","EstimatedSalary"]]  # independant value
Y = data["Purchased"] # Dependant value
#a=1

 
# Standardize values for Age and EsimatedSalary
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = pd.DataFrame( sc_x.fit_transform(X))

# Using standard ML librares . SKLearn
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.15, random_state = 0) 
# Note: if the training set size more, accuracy increases. upto 95% . test_zie = 0.1. 
# when it is become 100% as well

#Create a model
reg = LogisticRegression()
# Train the model
reg = reg.fit(Xtrain,Ytrain)
# Reg stores both m and c values internally


# Predict 
Y_predict = reg.predict(Xtest)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest,Y_predict)
print ("Confusion Matrics:\n", cm)
#Notes: Accuracy falls to 68% when we do not standarize the values.

# Notes: When normize and without  gender information. Accuracy increases to 89%
#Confusion Matrics:
# [[65  3]
# [ 8 24]]
#Accuracy :
# 0.89

# Notes: When normize and gender information. Accuracy increases to 91%
#Confusion Matrics:
# [[65  3]
# [ 6 26]]
#Accuracy :
# 0.91

from sklearn.metrics import accuracy_score
acc = accuracy_score(Ytest,Y_predict)
print("Accuracy :\n",acc)


R2_score = reg.score(Xtest,Ytest) # score internally calcualtes the predicted values . 
print(R2_score)


# Data visualization.
from matplotlib.colors import ListedColormap

X_set, y_set = np.array(Xtest), np.array(Ytest) 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,  
                               stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01))     
    

import matplotlib.pyplot as plt

plt.contourf(X1, X2, reg.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
      

plt.title('Classifier (Test set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show() 







