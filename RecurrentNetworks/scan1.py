# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:40:39 2020

@author: Guru Prasad Muppana

Theano scan functionality
scan(
    fn,
    sequences=None,
    outputs_info=None,
    non_sequences=None,
    n_steps=None,
    truncate_gradient=-1,
    go_backwards=False,
    mode=None,
    name=None,
    profile=False,
    allow_gc=None,
    strict=False)

scan is used for calling function multiple times over a list of values, the function may contain state.

"""

import numpy as np
import theano
import theano.tensor as T

input_seq = [1,2,3,4,5,6,7,8,9,11] # 

# Funcitons that is called from the scan operation.
def square(x):
    return x*x

s_x = T.ivector()

s_y, _ = theano.scan(
       fn = square,
       sequences = s_x,
       #n_steps=s_x.shape[0]
       )

squars = theano.function(
        inputs=[s_x],
        outputs=[s_y]
        )

y = squars(input_seq) 
print(squars(input_seq))

# sum of two array:

x1 = T.ivector()
x2 = T.ivector()
#y = T.ivector()

def vector_plus (x1,x2):
    return x1+x2

y,_ = theano.scan(
        fn = vector_plus,
        sequences=[x1,x2])

x1_x2_sum = theano.function(
        inputs = [x1,x2],
        outputs = y
        )


first_vector = [1,2,3,4,5]
second_vector = [10,20,30,40,50]
print(vector_plus(first_vector,second_vector))
print(x1_x2_sum(first_vector,second_vector))


print("_______________")

s_x = T.ivector()
v_sum = theano.shared(np.int32(0))
s_y, update_sum = theano.scan(
    fn= lambda x,y:x+y,
    sequences = [s_x],
    outputs_info = v_sum)
fn = theano.function([s_x], s_y, updates=update_sum)
  
print(v_sum.get_value()) # 0
print(fn([1,2,3,4,5])) # [1,3,6,10,15]
print(v_sum.get_value()) # 15
print(fn([-1,-2,-3,-4,-5])) # [14,12,9,5,0]
print(v_sum.get_value()) # 0


s_x0 = T.iscalar()
s_x1 = T.iscalar()
s_n = T.iscalar("s_n")
s_y, _ = theano.scan(
    fn = lambda x1,x2: x1+x2,
    outputs_info = [dict(initial=T.join(0,[s_x0, s_x1]), taps=[-2,-1])],
    n_steps = s_n

)
fn_fib = theano.function([s_x0, s_x1, s_n], s_y)
print(fn_fib(1,1,10))
# [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]













