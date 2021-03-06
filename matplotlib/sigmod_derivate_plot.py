# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:06:37 2020

@author: Guru Prasad Muppana

Plot sigmod and its derivate
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmod(x):
    return 1/(1+np.exp(-x))

n=-10
x = np.linspace(-n,n,1000,endpoint=False)

y = sigmod(x)
plt.plot(x,y, c="r", label="sigmod")
plt.xlabel("x")
plt.ylabel("y = sigmod(x)")
plt.legend()
plt.show()

# derivate of sigmod'(x)
y_hat = sigmod(x)*(1-sigmod(x))

plt.plot(x,y_hat, c="b", label="derivate of sigmod")
plt.xlabel("x")
plt.ylabel("y = sigmod derivate(x)")
plt.legend()
plt.show()

plt.plot(x,y, c="g", label="sigmod'")
plt.plot(x,y_hat, c="b", label="derivate of sigmod")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#y2 = x*(1-x)
#
#plt.plot(x,y2, c="y", label="x*(1-x)")
#plt.grid()
#plt.xlabel("x")
#plt.ylabel("y=x*(1-x)")
#plt.legend()
#plt.show()
#print(y[-1])
#




