import numpy as np
import random

def sigmoid(x):
    return 1/(np.exp(-np.array(x))+1)

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class ANN(object):

    def __init__(self,layers):
        """
        layers should be an array of the number of nodes in each layer.
        The length of layers determines the number of layers.

        by default the activation function is the sigmoid and the cost function
        is squared-error

        We randomly generate our initial biases and weights using a standard
        normal distribution.
        """
        self.num_layers = len(layers)
        self.layers = layers

        self.bias = [np.random.randn(y, 1) for y in layers[1:]]
        """
        self.weights returns an array of matrices; self.weights[i] returns
        the weight matrix between layers i+1 and i+2, where the k-th column
        contains the weights corresponding to the k-th neuron in layer i+1
        """
        self.weights = [np.random.randn(y,x) for x,y in zip(layers[:-1], layers[1:])]

    def reset_weights(self):
        self.bias = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.layers[:-1], self.layers[1:])]

    def evaluate(self,a):
        """
        returns the output of the network for given input array "a"
        """
        for b,w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def backprop(self,x,y):
        """"
        performs backpropogation based on a sample point, x, and the
        corresponding desired output, y;

        explicitly, we compute the "error derivatives" for the weights and biases,
        which we return as a tuple (grad_b,grad_w);

        these will then be averaged and scaled in the train loop, and the network's
        weights/bias will be adjusted accordingly
        """
        grad_b = [np.zeros(b.shape) for b in self.bias]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        a_l = np.array(x)
        a = [a_l]
        zs = []
        Ds = []
        for b,w in zip(self.bias, self.weights):
            z_l = np.dot(w,a_l)+b
            zs.append(z_l)
            a_l = sigmoid(z_l)
            a.append(a_l)
            d_l = np.diag(dsigmoid(z_l).squeeze())
            Ds.append(d_l)
        delta_l = np.dot(Ds[-1],(a[-1]-y))
        grad_b[-1] = delta_l
        grad_w[-1] = np.dot(delta_l, a[-2].transpose())
        for k in range(2,self.num_layers):
            delta_l = np.dot(np.dot(Ds[-k],self.weights[-k+1].transpose()), delta_l)
            grad_b[-k] = delta_l
            grad_w[-k] = np.dot(delta_l, a[-k-1].transpose())
        return (grad_b,grad_w)

    def train(self,X,Y,m,eta,N):
        """
        Performs SGD to train the network

        X = training data, stored as n1 x n2 array, where each column is a data
            point in R^n1 and we have n2 many points
        Y = desired output matrix; for classification should be
            num_classes x n2
        m = sample size at each step of SGD
        eta = step size(s)
        N = max number of iterations/ steps
        """
        start = time.time()
        num_col = X.shape[1]
        count = 0
        while count <= N:
            #eta = eta*(N-2*count)/N
            sample = random.choices(range(num_col),k=m)
            xs = X[:,sample]
            ys = Y[:,sample]
            delta_b = [np.zeros(b.shape) for b in self.bias]
            delta_w = [np.zeros(w.shape) for w in self.weights]
            for i in range(m):
                x = np.array([xs[:,i]]).transpose()
                y = np.array([ys[:,i]]).transpose()
                delta_bi, delta_wi = self.backprop(x,y)
                delta_b = [db+dbi for db, dbi in zip(delta_b, delta_bi)]
                delta_w = [dw+dwi for dw, dwi in zip(delta_w, delta_wi)]
            self.weights = [w-(eta/m)*dw
                        for w, dw in zip(self.weights, delta_w)]
            self.bias = [b-(eta/m)*db
                        for b, db in zip(self.bias, delta_b)]
            count += 1
        end = time.time()
        print('Training Complete!')
        print('Training took:',end-start, 'seconds')

    def test(self,X,Y):
        num_col = X.shape[1]
        answers = []
        for i in range(num_col):
            x = np.array([X[:,i]]).transpose()
            y = np.array([Y[:,i]]).transpose()
            pred = convert_pred(self.evaluate(x))
            if np.array_equal(y,pred):
                answers.append(1)
            else:
                answers.append(0)
        num_right = np.count_nonzero(answers)
        print('Correctly labeled:', num_right, 'out of', num_col)
        print('Percent correct:')
        return np.mean(answers)
