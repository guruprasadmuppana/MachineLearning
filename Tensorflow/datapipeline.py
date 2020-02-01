# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:25:31 2019

@author: Admin
"""

# Data pipeline example

import numpy as np
import tensorflow as tf

x_input = np.random.sample((1,2))
print(x_input)


# using a placeholder
x = tf.placeholder(tf.float32, shape=[1,2],name="x")


# Setting up data pipeline. This is done by from tensor slices in Dataset
# pass the source of data variable.
dataset = tf.data.Dataset.from_tensor_slices(x)
# initialize the iterator
iterator = dataset.make_initializable_iterator() # it is getting deprecated.
# move the cursor to the first (next) iterator
get_next = iterator.get_next()

# Execute the operation
with tf.Session() as session:
    # iterator is like any other node. Hence you can use the node in the run command
    # pass node along with input values. Note that it is setup step.
    # You need to go to  the next value in by moving the cursor.
    session.run(iterator.initializer,feed_dict={x:x_input})
    # get the first value in the 
    print (session.run(get_next))


