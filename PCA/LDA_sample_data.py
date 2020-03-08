# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:11:33 2020

@author: Muppana Guru Prasad 

LDA with a simple data set

https://www.youtube.com/watch?v=mtTVXZq-9gE&t=108s


"""
import numpy as np

#c1 = np.array([(4,1),(2,4),(2,3),(3,6),(4,4)])
#c2 = np.array([(9,10),(6,8),(9,5),(8,7),(10,8)])

c1 = np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
c2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

mu1 = np.mean(c1, axis=0,keepdims=True)
mu2 = np.mean(c2,axis=0,keepdims=True)
print ("mu1, mu2",mu1,mu2)

d1 = c1 - mu1
d2 = c2 - mu2
mu = mu1 - mu2

if True:
    print("Hello")

cov_1 = (np.matmul(d1.T,d1))/len(c1)
cov_2 = (np.matmul(d2.T,d2))/len(c2)
cov_w = cov_1 + cov_2

mu_w = np.matmul(mu.T,mu)

cov_w_inv = np.linalg.inv(cov_w)
cov_w_inv_T = cov_w_inv.T

eigen_vector = np.matmul(mu,cov_w_inv_T)
print(eigen_vector)


#cov1 = np.cov(c1)
#cov2 = np.cov(c2)
#
#cov = cov1 + cov2
#
#print(cov)




## Python code to demonstrate the  
## use of numpy.cov 
#import numpy as np 
#  
#x = [1.23, 2.12, 3.34, 4.5] 
#  
#y = [2.56, 2.89, 3.76, 3.95] 
#  
## find out covariance with respect  rows 
#cov_mat = np.stack((x, y), axis = 1)  
#  
#print("shape of matrix x and y:", np.shape(cov_mat)) 
#  
#print("shape of covariance matrix:", np.shape(np.cov(cov_mat))) 
#  
#print(np.cov(cov_mat)) 
