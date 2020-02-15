import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.utils import shuffle


def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    # Download this file from Kaggle
    #https://www.kaggle.com/c/digit-recognizer
    train = pd.read_csv('mnist_train.csv').values.astype(np.float32)
    train = shuffle(train)

    split_amount = 1000
    Xtrain = train[:-split_amount,1:] / 255
    Ytrain = train[:-split_amount,0].astype(np.int32)

    Xtest  = train[-split_amount:,1:] / 255
    Ytest  = train[-split_amount:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    pca = PCA()
    
#    reduced = pca.fit_transform(Xtrain)
#    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=0.5)
    
    reduced = pca.fit_transform(Xtest)
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytest, alpha=0.5)
    
    # Note that PCA is not meant for classification but to reduce the dimensitional 
    # at the sacrificing a bit of quality of information.
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.show()



if __name__ == '__main__':
    main()