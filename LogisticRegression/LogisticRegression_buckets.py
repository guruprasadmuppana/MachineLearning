# -*- coding: utf-8 -*-
"""


@author: Guru Prasad Mupppana

Binary classification.

"""

import numpy as np
import pandas as pd

# Read the file headbrain.cvs file.
filename = "number_positives_negatives.xlsx"

data = pd.read_excel(filename)

X = data["X"]  # independant value
Y = data["Y"] # Dependant value


day = pd.get_dummies(data["Y"],drop_first=False)
data.drop(["Y"],axis=1,inplace=True)
data = pd.concat([data,day],axis=1)
Y = data[day.columns] # Dependant value

Y=np.array(Y)
Y = np.argmax(Y,axis=1) 

X = np.array(X)
# Standardize values for Age and EsimatedSalary

# Using standard ML librares . SKLearn
from sklearn.linear_model import LogisticRegression 
#from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.20, random_state = 0) 
# Note: test size is fine tune at 20% with 99.888 %accuracry

#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#
#sc_x.fit(Xtrain)
## Apply transform to both the training set and the test set.
#Xtrain = sc_x.transform(Xtrain)
#Xtest = sc_x.transform(Xtest)

#X = pd.DataFrame( sc_x.fit_transform(X))

#Create a model
#reg = LogisticRegression()

reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

#reg = LogisticRegression(solver = 'lbfgs')
#model.fit(train_img, train_lbl)
 
# Train the model
reg = reg.fit(Xtrain.reshape(-1,1),Ytrain)
# Reg stores both m and c values internally


# Predict 
Y_predict = reg.predict(Xtest.reshape(-1,1))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest,Y_predict)
print ("Confusion Matrics:\n", cm)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Ytest,Y_predict)
print("Accuracy :\n",acc)

print(Xtest[Ytest!=Y_predict])
print(Ytest[Ytest!=Y_predict])
print(Y_predict[Ytest!=Y_predict])
print( reg.predict(np.array(-9.99).reshape(1,-1)))
print( reg.predict(np.array(123.99).reshape(1,-1)))

#
#R2_score = reg.score(X,Y) # score internally calcualtes the predicted values . 
#print(R2_score)






