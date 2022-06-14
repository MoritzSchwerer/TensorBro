#!/usr/bin/env python3

from .base import Layer
import numpy as np

class Dense(Layer):
    """
    A Fully connected layer for a nural network

    Atributes:
    ----------
    X : matrix(n0, m)
        the input to the layer that will be passed through
    W : matrix(n1,n0)
        the weight matrix that will generate the Z values for a given input
    B : matrix(N1, 1)
        the bias that will be added after the weights have been applied
    optimizer : Optimizer
        the optimizer will handle the backward propagation of the gradients


    Methods:
    --------
    forward(X):
        takes the input and generates the Z values to the layer

    backward(dZ):
        takes the gradient and returns the correct new W and B
    """
    def __init__(self, input_size=1, output_size=1, optimizer=[]):
        self.W = np.random.randn(output_size, input_size)
        self.B = np.random.randn(output_size, 1)
        self.optimizer = optimizer

    def forward(self, X):
        """
        Paramters
        ---------
        X : matrix(float)
            the input to this layers forward computation

        returns the nural calculation result for the current layer
        """
        self.X = X
        return np.matmul(self.W, X) + self.B

    def backward(self, dZ):
        """
        Parameters
        ----------
        dZ : matrix(float)
            the error of this layers values

        returns the error for the previous layers backward step
        """
        self.W, self.B = self.optimizer.optimize(dZ, self.X, self.W, self.B)
        return np.matmul(self.W.T, dZ) # dA-1
