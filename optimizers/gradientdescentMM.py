#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod
from .optimizer import Optimizer

class GradientDescentMM(Optimizer):
    """
    An implementation of an improved version of  Gradient Descent that additionaly uses momentum

    Attributes
    ----------
    lr : number
        the learning rate for the optimizer
    sdW : matrix(number)
        the accumulated gradient for W
    sdB : matrix(number)
        the accumulated gradient for B
    beta : number
        the factor that decides how much impact the newly calculated dW has on sdW

    Method
    ------
    optimize(dZ, X, W, B)
        does the backwards calculation for the layer in the nural network and returns the new W and B
    """
    def __init__(self, input_size, output_size, learning_rate=0.1, beta=0.9):
        self.lr = learning_rate
        self.sdW = np.zeros((output_size, input_size))
        self.sdB = np.zeros((output_size, 1))
        self.beta = beta

    def optimize(self, dZ, X, W, B):
        """
        Calculates the backward step for the neural network and returns the updated
        weights and biases

        Parameters
        ----------
        dZ : matrix(number)
            the gradient for Z
        X : matrix(number)
            the input to the layer
        W : matrix(number)
            the weights of the layer
        B : matrix(number)
            the bias of the vector
        """
        dW = np.matmul(dZ, X.T) #/ dZ.shape[1]
        dB = np.sum(dZ, axis=1, keepdims=True) #/ dZ.shape[1]
        self.sdW = self.sdW * self.beta + (1 - self.beta) * dW
        self.sdB = self.sdB * self.beta + (1-self.beta) * dB
        nW = W - self.lr * self.sdW
        nB = B - self.lr * self.sdB
        return nW, nB
