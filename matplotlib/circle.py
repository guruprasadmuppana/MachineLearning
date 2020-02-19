# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:40:35 2020

@author: Guru Prasad Muppana

circle

"""

import matplotlib.pyplot as plt

def create_circle(r,fill=False):
	circle= plt.Circle((0,0), radius= r, fill=fill)
	return circle

def show_shape(patch):
	ax=plt.gca()
	ax.add_patch(patch)

c= create_circle(10)
c2=create_circle(5,fill=True)
show_shape(c)
show_shape(c2)

plt.axis('scaled')
plt.show()



