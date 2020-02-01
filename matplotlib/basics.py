# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 08:08:38 2019

@author: Admin

https://matplotlib.org

Modules:
        pyplot
        pylab # deprecated.
        
        mplot3d 

        Streamplot


"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

a = pd.DataFrame(np.random.rand(4,5),columns=list("abcde"))  
a_asarray = a.values

b = np.matrix([
            [1,2],
            [3,4]
            ])
b_asarray = np.asarray(b)       

#print(a)
#print(b)  
#print("\n")
#print(a_asarray)
#print(b_asarray) 


# Figure is a top level object. There is a default figure provided by matplotlib
#fig = plt.Figure()
#fig.suptitle('Hello figure !')
#
#fig, ax_lst = plt.subplots(2,2)



#x = np.linspace(0,3,100) # creates 100 points between 0 and 2. i.e 98 new points will be created.
## Using the above points on x, y is caculated using x, x**2, and x**3 curves
## We will use this for ploting. plot will draw (line) plot.
## There is already one default figure. 
#plt.plot(x,x, label="Linear")
#plt.plot(x,x*x,label="Quadritic")
#plt.plot(x,x**3,label="cubic")
#plt.xlabel("x-label")
#plt.ylabel("y- label")
#plt.title("Curves with varing degrees")
#plt.legend()
#plt.show()



## Showing sin wave.
#x = np.arange(0,10,0.2) # between 0 and 10, with step- 0.2
#y = np.sin(x)
#fig , ax_lst = plt.subplots() # creaes 1x1 sub-plot
#ax_lst.plot(x,y)
#plt.show()


#x = np.arange(0,10,0.2) # between 0 and 10, with step- 0.2
#y = np.sin(x)
#fig , (ax1,ax2,ax3) = plt.subplots(1,3)
#ax1.plot(x,y)
#data1_x, data1_y = np.random.randn(2,100)
#data2_x, data2_y = np.random.randn(2,100)
#ax2.plot(data1_x,data1_y,**{'marker': 'o'})
#plt.xlabel("2nd axis") # not getting printed
#ax3.plot(data2_x,data2_y, **{'marker': 'x'})
#plt.xlabel("3rd axis")
#plt.show()


#plt.ion()  # not getting shown while running from the command prompt using python code.
## ion -> interactive on. plt.ioff
#plt.plot([1.6, 2.7])
#plt.title("test")
#plt.xlabel("X axis")
#plt.show()
#
#plt.ioff()
#for i in range(3):
#    plt.plot(np.random.rand(10))
#    plt.show()
#    
#






#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#
## Setup, and create the data to plot
#y = np.random.rand(100000)
#y[50000:] *= 2
#y[np.logspace(1, np.log10(50000), 400).astype(int)] = -1
#mpl.rcParams['path.simplify'] = True
#
#mpl.rcParams['path.simplify_threshold'] = 0.0
#plt.plot(y)
#plt.show()
#
#mpl.rcParams['path.simplify_threshold'] = 1.0
#plt.plot(y)
#plt.show()    



#
#
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.rcParams['path.simplify_threshold'] = 1.0
#
## Setup, and create the data to plot
#y = np.random.rand(100000)
#y[50000:] *= 2
#y[np.logspace(1,np.log10(50000), 400).astype(int)] = -1
#mpl.rcParams['path.simplify'] = True
#
#mpl.rcParams['agg.path.chunksize'] = 0
#plt.plot(y)
#plt.show()
#
#mpl.rcParams['agg.path.chunksize'] = 10000
#plt.plot(y)
#plt.show()
##
#import matplotlib.style as mplstyle
#mplstyle.use('fast')


#
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'b^')
#plt.plot([1, 2, 3, 4], [1,8, 27, 64], 'ro')
#plt.axis([0, 8, 0, 65]) # Axis limits
#plt.show()
#
##import numpy as np
## evenly sampled time at 200ms intervals
#t = np.arange(0., 5., 0.2)
## red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')  # All lines in one go.
#plt.show()




#data = {'a': np.arange(50),
#        'c': np.random.randint(0, 50, 50),
#        'd': np.random.randn(50)}
#data['b'] = data['a'] + 10 * np.random.randn(50)
#data['d'] = np.abs(data['d']) * 200
#
##plt.scatter('a', 'b', c='c', s='d', data=data)
#
##plt.scatter('a','b',data=data)
#
#plt.scatter('a','b',c='c',s='d',data=data)
## c= color
## s = size
#
#plt.xlabel('entry a')
#plt.ylabel('entry b')
#plt.show()


#
#names = ['group_a', 'group_b', 'group_c']
#values = [1, 25, 100]
#
#plt.figure(figsize=(12, 3))
#
#plt.subplot(131)
#plt.bar(names, values)
#
#plt.subplot(132)
#plt.scatter(names, values)
#
#plt.subplot(133)
#plt.plot(names, values)
#
#
#
#plt.suptitle('Categorical Plotting')
#plt.show()


#line, = plt.plot(x, y, '-')
#line.set_antialiased(False) # turn off antialiasing
#
#x1, y1 = 1,1
#x2,y2 = 5,5
#
#lines = plt.plot(x1, y1, x2, y2)
#plt.setp(lines, color='r', linewidth=5.0)  # this is not working.
## use keyword args
## or MATLAB style string value pairs
##plt.setp(lines, 'color', 'r', 'linewidth', 2.0)


#
#
#def f(t):
#    return np.exp(-t) * np.cos(2*np.pi*t)
#
#t1 = np.arange(0.0, 5.0, 0.1)
#t2 = np.arange(0.0, 5.0, 0.02)
#
#plt.figure()
#plt.subplot(211)
#plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
#plt.subplot(212)
#plt.plot(t2, np.cos(2*np.pi*t2), 'r.') # r-- ; r-.
#plt.show()
#


ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

plt.show()