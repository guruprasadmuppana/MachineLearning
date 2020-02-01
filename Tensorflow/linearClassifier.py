# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:08:10 2019

@author: Admin
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from six.moves import urllib


import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


from IPython.display import clear_output

# Load the titanic passenger servival data. 
# Get the files from net.
# https://storage.googleapis.com/tf-datasets/titanic/train.csv
# https://storage.googleapis.com/tf-datasets/titanic/eval.csv

train_file = "./titanic/train.csv"
test_file = "./titanic/eval.csv"

train_data = pd.read_csv(train_file) # returns the dataframe
test_data = pd.read_csv(test_file) # returns the dataframe

# collectign the column of values for servival column.
#y_train = train_data["survived"]
#y_test = test_data["survived"]

y_train = train_data.pop("survived")
y_test = test_data.pop("survived")



# Inspecting the data 
#train_data.shape
#train_data.head()
#train_data.describe()

#train_data.shape[0], test_data.shape[0]

# train_data.age.hist(bins=20)
# Observation: Major people between 25 and 30 with mean value 29.

# train_data.sex.value_counts().plot("bar") # display vertial hist
# train_data.sex.value_counts().plot("barh") # displays horizontal hist
# obseveration : Male are more than female. Almost twice.

# train_data.sex[train_data.sex == "male"].count()/train_data.sex.count()
# 65 % of people are male.

# train_data["class"].value_counts().plot("barh")
# Observation: Majority of people third class

# pd.concat([train_data, y_train], axis=1)
#           .groupby('sex').survived.mean()
#            .plot(kind='barh').set_xlabel('% survive')


# Finding unitque values
# train_data["class"].value_counts() # gives category values for each unique value.
# train_data.sex.unique() gives various unique values


# Feature engineering
# Selecting and crafting the right set of feature columns is key to learning an effective model.

#The linear estimator uses both numeric and categorical features. Feature columns work with all 
#TensorFlow estimators and their purpose is to define the features used for modeling. 
#Additionally, they provide some feature engineering capabilities 
#like one-hot-encoding, normalization, and bucketization

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = train_data[feature_name].unique()
    feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary)
    feature_columns.append(feature_col)

for feature_name in NUMERIC_COLUMNS:
    feature_col = tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
    feature_columns.append(feature_col)



# input function. The source of data. It can be DataFrame, File,
# Do we need inner function in teh code below?
#def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
#  def input_function():
#      ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
#      if shuffle:
#          ds = ds.shuffle(100) # reduced from 1000 to 100.
#          #ds = ds.batch(batch_size).repeat(num_epochs) # what does it do?
#          ds = ds.repeat(num_epochs).batch(batch_size) # what does it do?
# 
#      return ds #tf.compat.v1.data.make_one_shot_iterator(ds) #.batch(batch_size)
#  return input_function
#
#train_input_fn = make_input_fn(train_data, y_train)
#eval_input_fn = make_input_fn(test_data, y_test, num_epochs=1, shuffle=False)
#
#ds = make_input_fn(train_data, y_train, batch_size=10)() # it is a functional call.
#



#i=1
#for feature_batch, label_batch in ds.take(1):
##  print('Some feature keys:', list(feature_batch.keys()))
##  print()
## #print('A batch of class:', feature_batch['class'].numpy())
##  print()
##  print('A batch of Labels:', label_batch)
#    i += 1
#    print (i)

#
#age_column = feature_columns[7]
#tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()

linear_est = tf.estimator.LinearClassifier(n_classes=2,
                                           model_dir="./titanic/model",
                                           feature_columns=feature_columns)


FEATURES=['sex', 'n_siblings_spouses','parch', 'class','deck','embark_town', 'alone','age','fare']
LABEL = "survived"


def get_input_fn(data_set, label_set, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k:data_set[k].values for k in FEATURES}),
            y=pd.Series(label_set),
            batch_size = n_batch,
            num_epochs = num_epochs,
            shuffle=shuffle)
    


linear_est.train(input_fn=get_input_fn(train_data,y_train,
                                       num_epochs=1000,
                                       n_batch=10,
                                       shuffle=True),
    steps=1000)
    
    
    
result = linear_est.evaluate(input_fn=get_input_fn(test_data,y_test,
                                       num_epochs=1,
                                       n_batch=10,
                                       shuffle=False),
    steps=1000)



result1 = list(linear_est.predict(input_fn=get_input_fn(test_data,y_test,
                                       num_epochs=1,
                                       n_batch=10,
                                       shuffle=False)))
     
#pred = list(linear_est.predict)
#
#from sklearn.metrices import classification_report
#print(classification_report(y_test))

clear_output()
print(result)

print(result1)

gender = []
m =""
count = 0
total = 0
for k in result1:
    total +=1
    if ( k["probabilities"][1] > 0.5):
        gender.append(test_data[total])
        m = "male"
    else:
        m = "female"
        gender.append("SSS")
    

print("F",count)
print("T",total)

print(gender)
    
    

