
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


def computeWeightsUsingGradientDeceint(X,Y):

    N = len(X)
    baisOneColumn = np.ones(N)
   
    #np.concatenate((X,baisOneColumn),axis=1)
    #P = np.hstack([X,baisOneColumn]).reshape(-1)
    X = np.c_[baisOneColumn,X]
    N,D = X.shape

    #Gradient descent
    costs = []
    w = np.random.randn(D) / np.sqrt(D) #  w0 and w1 (w0 is the bais)
#    w = [10,4]  # actual solution.
    learning_rate = 0.000001 # try multiple values: 0.1, 0.01, 0.001, 0.0001
    for t in range(100):
      # update w
      Yhat = X.dot(w)  # forward function
      delta = Yhat - Y # differce from the actual and predictions. i.e variance 
      w = w - learning_rate*X.T.dot(delta)  # Update the new wieghts. 
      # w <- w -alpha*d(J)/d(w)
    
      # Mean squared error i.e mean of error**2
      mse = delta.dot(delta) / N
      costs.append(mse)
    
        # plot the costs
    plt.plot(costs)
    plt.show()
    
    return w
     




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
    print(a,b)
    pY = predictY(X,a,b) # a and b can be part of class variable.
    DisplayData(X,pY,label= "Predicted",c="r")
    score(Y,pY)
    
    
    w = computeWeightsUsingGradientDeceint(X,Y)
    
    N = len(X)
    baisOneColumn = np.ones(N)
    X_one = np.c_[baisOneColumn,X]
    print("Final w",w)
    pY = X_one.dot(w) 
    #pY = predictY(X,w0,w1) # a and b can be part of class variable.
    DisplayData(X,pY,label= "Gradient Decent",c="b")
    score(Y,pY)
    
    