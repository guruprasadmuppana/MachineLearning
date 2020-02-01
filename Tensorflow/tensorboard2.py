# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:18:56 2019

@author: Admin
"""
from __future__ import print_function

import tensorflow as tf

# import MNIST data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

# parameters:
learning_rate=0.01
training_epochs = 25
batch_size =100
display_epoch = 1
logs_path = "./train/linreg/"

# Tf.Graph input
x = tf.placeholder(tf.float32, [None,784], name="InputData")
y = tf.placeholder(tf.float32,[None,10],name="LabelData")

# Set the model weights
W = tf.Variable(tf.zeros([784,10]), name="weights")
b = tf.Variable(tf.zeros([10]),name="Bais")

with tf.name_scope("model"):
    pred = tf.nn.softmax(tf.matmul(x,W)+b) # softmax ... for categorization.
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=[0] ,keepdims=True))
with tf.name_scope("SGD"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope("Accuracy"):
    acc = tf.equal(tf.arg_max(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))
    
# initialize the variables.
init  = tf.global_variables_initializer()


tf.summary.scalar("cost",cost)
tf.summary.scalar("accuracy",acc)
merged_summary_op = tf.summary.merge_all()

# Start the session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
    # Train the cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c,summary = sess.run([optimizer,cost,merged_summary_op], feed_dict = {x:batch_xs,y:batch_ys})
            summary_writer.add_summary(summary,epoch*total_batch+i)
            avg_cost += c/total_batch
        if (epoch+1) % display_epoch == 0:
            print("epoch","%04d" % (epoch+1),"cost","{:.9f}".format(avg_cost))
    print("Optimization finished")
    print("Accuracy",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
            
