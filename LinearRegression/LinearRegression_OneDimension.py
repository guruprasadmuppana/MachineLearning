
import numpy as np
import matplotlib.pyplot as plt

def GenerateOneDimesionData():
    N = 100
    with open('data_1d.csv', 'w') as f:
        X = np.random.uniform(low=0, high=100, size=N)
        Y = 4*X + 10 + np.random.normal(scale=5, size=N)
        for i in range(N):
            f.write("%s,%s\n" % (X[i], Y[i]))
    #    plt.scatter(X,Y,alpha=0.3, marker="^",cmap="green")


def LoadOneDimesionData():
    X = []
    Y = []
    for line in open('data_1d.csv'):
        x, y = line.split(',')
        X.append(float(x))
        Y.append(float(y))
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def DisplayData(X,Y,label="Line",c="green"):
    plt.scatter(X,Y,alpha=0.3, marker=r'$\clubsuit$',c=c, label=label)
    plt.title("X and Y values")    
    
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend(loc='upper left')
    plt.show()
    

def computeWieghts(X,Y):
    denominator = X.dot(X) - X.mean() * X.sum()
    a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
    b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
    return a,b 

def predictY(X,a,b):
    Yhat = a*X + b
    return Yhat

def score(Y,Yhat):
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    print("the r-squared is: {0}".format(r2))


if __name__ == "__main__":
    GenerateOneDimesionData()
    X, Y = LoadOneDimesionData()
    DisplayData(X,Y,label="Original")
    a,b = computeWieghts(X,Y)
    pY = predictY(X,a,b)
    DisplayData(X,pY,label= "Predicted",c="r")
    score(Y,pY)
