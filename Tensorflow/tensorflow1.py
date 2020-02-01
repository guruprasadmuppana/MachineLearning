# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:37:59 2019

@author: Admin
"""

# Tensorflow samples
# tensorflow multiplication of two numbers


import numpy as np
import tensorflow as tf

# 1 Step: Setting up a model. This is also called Graph.
# as it contains the nodes and edges. nodes are variable and operations.

# defining placeholders for tensor variables
x1 = tf.placeholder(tf.float32,name="x1")
x2 = tf.placeholder(tf.float32,name="x2")
# defininng the operation node using tf.
c = tf.multiply(x1,x2, name="c")


# supplying values to tensor scalar variables.
# passing list of values
x1_values = [1,2,3]
x2_values = [4,5,6]
# passing scalar values
a = 100
b = 200  
# passing the strings

# Creating a session to run the model

with tf.Session() as session:
    # run the model using run method on session
    #mulipling list of values
    result1 = session.run(c, feed_dict={x1:x1_values,x2:x2_values})
    #multiplying two scalar values
    result2 = session.run(c,feed_dict={x1:a,x2:b})
    # multiplying two matrices
    result3 = session.run(c,feed_dict={x1:[[1,2],[2,3]],x2:[[1,2],[3,4]]})
    print(result1)
    print(result2)
    print(result3)
    
    
data_placeholder_a= tf.placeholder(tf.float64)
power_a = tf.pow(data_placeholder_a, 3)
with tf.Session() as sess:  
    #data = np.random.rand(1, 10)
    data = [1,2,3,4,5]
    print(sess.run(power_a, feed_dict={data_placeholder_a: data}))  # Will succeed.    
    
    
# notes: There are two machanism to load the data.
    #1. Load the data in memory when the size is low. Say less then 10 GB assuming that you have 16 GB RAM
    #2. Tensorflow Data pipeline. This is used when the data is huge i.e it is more than you computer memory
    # This helps to use multiple CPUs to run the model 
    


