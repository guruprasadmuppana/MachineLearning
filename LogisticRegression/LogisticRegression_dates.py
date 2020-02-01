# -*- coding: utf-8 -*-
"""


@author: Guru Prasad Mupppana

Binary classification.

"""

import numpy as np
import pandas as pd

# Read the file headbrain.cvs file.
filename = "dates_train.xlsx"

data = pd.read_excel(filename)
a=1
day = pd.get_dummies(data["Day"],drop_first=False)
data.drop(["Day"],axis=1,inplace=True)
data = pd.concat([data,day],axis=1)


X = data[["DD","MM","YYYY"]]  # independant value
Y = data[day.columns] # Dependant value

Y=np.array(Y)

Y = np.argmax(Y,axis=1) 

a=1


 
# Standardize values for Age and EsimatedSalary

# Using standard ML librares . SKLearn
from sklearn.linear_model import LogisticRegression 
#from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.25, random_state = 0) 
# Note: if the training set size more, accuracy increases. upto 95% . test_zie = 0.1. 
# when it is become 100% as well

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

sc_x.fit(Xtrain)
# Apply transform to both the training set and the test set.
Xtrain = sc_x.transform(Xtrain)
Xtest = sc_x.transform(Xtest)

#X = pd.DataFrame( sc_x.fit_transform(X))

#Create a model
#reg = LogisticRegression()

reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

#reg = LogisticRegression(solver = 'lbfgs')
#model.fit(train_img, train_lbl)
 
# Train the model
reg = reg.fit(Xtrain,Ytrain)
# Reg stores both m and c values internally


# Predict 
Y_predict = reg.predict(Xtest)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest,Y_predict)
print ("Confusion Matrics:\n", cm)




from sklearn.metrics import accuracy_score
acc = accuracy_score(Ytest,Y_predict)
print("Accuracy :\n",acc)


R2_score = reg.score(Xtest,Ytest) # score internally calcualtes the predicted values . 
print(R2_score)






