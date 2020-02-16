
import numpy as np
import matplotlib.pyplot as plt


def GradientDecentOnCostFunction():
    lr = 1e-2
    x1 = 5
    x2 = -5
    
    def J(x1, x2): # Cost function with two variable
      return x1**2 + x2**4
    
    def g1(x1): # gradient of cost J with respect to x1. Derivtive d(J)/d(x1)
      return 2*x1
    
    def g2(x2): # gradient of cost J with respect to x2. Derivtive d(J)/d(x2)
      return 4*x2**3
    
    values = []
    for i in range(1000):
      values.append(J(x1, x2))
      x1 -= lr * g1(x1)
      x2 -= lr * g2(x2)
    
    values.append(J(x1, x2))
    
    print("(x1, x2)=",x1, x2)
    plt.plot(values)
    plt.show()


def GradientDecentUsingMatrices():

    N = 10
    D = 3
    X = np.zeros((N, D))  # b is added to X (x0,x1,x2)
    
    X[:,0] = 1 
    X[:5,1] = 1
    X[5:,2] = 1
    Y = np.array([0]*5 + [1]*5)
    
    print("X:", X)
    
    
    #Gradient descent
    costs = []
    w = np.random.randn(D) / np.sqrt(D) #  initialize w
    
    learning_rate = 0.001 # try multiple values: 0.1, 0.01, 0.001, 0.0001
    for t in range(1000):
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
    
    print("final w:", w)
    
    # plot prediction vs target
    plt.plot(Yhat, label='prediction')
    plt.plot(Y, label='target')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    GradientDecentOnCostFunction()
    GradientDecentUsingMatrices()
    