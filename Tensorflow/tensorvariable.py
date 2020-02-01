# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:14:09 2019

@author: Admin
"""
# Three import featurs of tensors.
#A unique label (name)
#A dimension (shape)
#A data type (dtype)

# Four types of tensors
#tf.Variable 
#tf.constant 
#tf.placeholder 
#tf.SparseTensor 

import tensorflow as tf

# rank 0
# default 

r1 = tf.constant(1,tf.int64)
print(r1)
r2 = tf.constant(1,tf.int64, name="x")
print(r2)
r3 = tf.constant(1.23,tf.float64,name="r3_decimal")
print(r3)
r4 = tf.constant("Hello Universe",tf.string,name="r4_string")
print(r4)

# rank 1
r5 = tf.constant([1,2,3],tf.int64,name="list_of_numbers")
print(r5)
r6 = tf.constant([True, False, True],tf.bool, name="bools")
print(r6)

#Rank 2

r2_matrix = tf.constant([ [1, 2],
                          [3, 4] ],tf.int16, name="mat")
print(r2_matrix)

# Shape of tensor
m_shape = tf.constant([ [10, 11],
                        [12, 13],
                        [14, 15] ] )                      
 	
print(m_shape.shape)


## Rank 3
r5_matrix = tf.constant([ [[1, 2],
                           [3, 4], 
                           [5, 6]] ], tf.int16)
print(r5_matrix)

## Rank 2: Watch for number of [] brackets
r6_matrix = tf.constant([ [1, 2],
                           [3, 4], 
                           [5, 6]], tf.int16)
print(r6_matrix)

# some useful vectors
print(tf.zeros(10))
print(tf.zeros([10,10]))
print(tf.ones([10,10,10]))

print(tf.ones(m_shape.shape))	


# Type cast
type_float = tf.constant(1.23,tf.float64)
type_int = tf.cast(type_float,tf.int32)
print(type_float.dtype)
print(type_int.dtype)

# Creating operator
sqrt = tf.constant([2.0],dtype=tf.float64)
print(tf.sqrt(sqrt))

#tf.add(a, b) 
#tf.substract(a, b) 
#tf.multiply(a, b) 
#tf.div(a, b) 
#tf.pow(a, b) 
#tf.exp(a) 
#tf.sqrt(a) 

# Add
tensor_a = tf.constant([[1,2]], dtype = tf.int32)
tensor_b = tf.constant([[3, 4]], dtype = tf.int32)

tensor_add = tf.add(tensor_a, tensor_b)
print(tensor_add)


# Variables
#tf.get_variable(name = "", values, dtype, initializer)
# initializer can be used to initial with all zeros or null etc.

var_init_1 = tf.get_variable("var_init36", [1, 2], dtype=tf.int32,initializer=tf.zeros_initializer) # zeros are used for initialization
print(var_init_1.shape)	
print(var_init_1)

tensor_cont = tf.constant([[10,20],[30,40]],dtype=tf.int32,name="tensor_matric")

var = tf.get_variable(name="vara2" , dtype=tf.int32,initializer=tensor_cont)
print(var)
    	
#tf.placeholder(dtype,shape=None,name=None )





