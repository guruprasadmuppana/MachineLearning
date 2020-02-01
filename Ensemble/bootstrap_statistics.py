# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:00:31 2020

@author: Guru Prasad Muppana

Bootstrap - concepts:
    If you take out  samples with replacement from a given set X, the mean/std/etc of X will be similar
    the mean/std/etc of samples taken out of set X

"""

#imports
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

B = 500 # boot strap samples.
N = 20 # sample size main sample set X

X = np.random.randn(N)

x_mean = X.mean()
x_std = X.std()

#boot strap samples:

individual_samples = np.empty(B) # initize

stds = np.empty(B)

for b in range(B):
    sample = np.random.choice(X,size=N) # get the points from X but with replacement.
    # note that original set X and sample set are similar size . Hence,the points taken from X will duplicate
    individual_samples[b] = sample.mean() # store the mean of individual bootstrap samples
    stds[b] = sample.std()
    
bootstrap_mean = individual_samples.mean()
bootstrap_std = individual_samples.std()  #

bootstrap_std_mean = stds.mean()/np.sqrt(B)  #



#
plt.hist(individual_samples,bins=12)
plt.axvline(x=x_mean,color="yellow",linestyle="--",label="X mean")
plt.axvline(x=bootstrap_mean,color="yellow",linestyle="--",label="X mean")
#plt.show()


# display PPT with 0.25% to 95%

lower = bootstrap_mean + norm.ppf(0.025)*bootstrap_std
upper = bootstrap_mean + norm.ppf(0.975)*bootstrap_std

#lower = bootstrap_mean + norm.ppf(0.025)*bootstrap_std_mean
#upper = bootstrap_mean + norm.ppf(0.975)*bootstrap_std_mean

lower_X = x_mean + norm.ppf(0.025)*x_std/np.sqrt(N) # why is this required ?
upper_X = x_mean + norm.ppf(0.975)*x_std/np.sqrt(N)

#
#lower_X = x_mean + norm.ppf(0.025)*x_std 
#upper_X = x_mean + norm.ppf(0.975)*x_std

plt.axvline(x=lower,color="g",linestyle="--",label="lower - bootstrap")
plt.axvline(x=upper,color="g",linestyle="--",label="upper - bootstrap")

plt.axvline(x=lower_X,color="r",linestyle="--",label="lower - X")
plt.axvline(x=upper_X,color="r",linestyle="--",label="upper - X")
plt.legend()

plt.show()


