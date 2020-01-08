import numpy as np

def ReLU(x):
    return np.array([max(0,x) for x in x])

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def squared_error(X,Y):
    "returns squared error, rescaled by 1/2 for convenience"
    return 0.5*np.mean(np.square(X-Y))
