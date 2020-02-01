# -*- coding: utf-8 -*-
"""


@author: Guru Prasad Mupppana

MNIST

http://yann.lecun.com/exdb/mnist/

train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

"""

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata

from mnist import MNIST



# Read the file headbrain.cvs file.
filename = "User_Data.csv"

filename_img = "../Datasets/mnist/t10k-images-idx3-ubyte"
filename_lbl = "../Datasets/mnist/t10k-labels-idx1-ubyte"

a=1
data_home = "D:/python/Guru/LogisticRegression/data/"
#mnist = fetch_mldata(data_home)

#mndata = MNIST(data_home)
#
#
#mndata.test_img_fname = 't10k-images.idx3-ubyte'
#mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
#
#
#mndata.train_img_fname = 't10k-images.idx3-ubyte'
#mndata.train_lbl_fname = 't10k-labels.idx1-ubyte'


#images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()

#################################
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# You can add the parameter data_home to wherever to where you want to download your data
#mnist = fetch_mldata('MNIST original')

mndata = MNIST(data_home)

mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

#images, labels = mndata.load_training()
images, labels = mndata.load_testing()


#print(mndata.data.shape)
#print(mndata.COL_NAMES)
#print(mndata.target.shape)
#print(np.unique(mndata.target))

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(
    images, labels, test_size=.25, random_state=122)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#model = LogisticRegression(solver = 'lbfgs')
model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

model.fit(train_img, train_lbl)

# use the model to make predictions with the test data
y_pred = model.predict(test_img)
# how did our model perform?
count_misclassified = (test_lbl != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_lbl, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

t= test_img[test_lbl != y_pred]



test_lbl = np.array(test_lbl) # Key conversion

lbl = test_lbl[(test_lbl != y_pred)]
pred = y_pred[(test_lbl != y_pred)]


import matplotlib.pyplot as plt




fig = plt.figure()
image_size = 28
for i in range(5):
    for j in range(5):
        a = fig.add_subplot(5, 5, i*5+j+1)
        img = np.array(t[i*5+j]);
        #print(i*5+j)
        img = img.reshape(image_size,image_size)
        #image =np.asarray(images[0]).squeeze()
        imgplot = plt.imshow(img)
        imgplot.set_clim(0.0, 0.7)
        a.set_title(str(lbl[i*5+j])+","+str(pred[i*5+j]))

plt.show()






