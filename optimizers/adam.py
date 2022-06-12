#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod
from .optimizer import Optimizer

class Adam(Optimizer):
    """
    An implementation of an improved version of  Gradient Descent that additionaly uses momentum

    Attributes
    ----------
    lr : number
        the learning rate for the optimizer
    vdW : matrix(number)
        momentum adjusted gradient for W
    vdB : matrix(number)
        momentum adjusted gradient for B
    sdW : matrix(number)
        exponential average gradient for W
    sdB : matrix(number)
        exponential average gradient for B
    beta1 : number
        the factor that decides how much impact the newly calculated dW has on vdW
    beta2 : number
        the factor that decides how much impact the newly calculated dW has on sdW
    epsilon : number
        constant optimization pararmeter for backward step

    Method
    ------
    optimize(dZ, X, W, B)
        does the backwards calculation for the layer in the nural network and returns the new W and B
    """
    def __init__(self, input_size, output_size, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=0.00000001):
        self.lr = learning_rate
        self.sdW = np.zeros((output_size, input_size))
        self.sdB = np.zeros((output_size, 1))
        self.vdW = np.zeros((output_size, input_size))
        self.vdB = np.zeros((output_size, 1))
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 1

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
        # calculate gradients
        dW = np.matmul(dZ, X.T) #/ dZ.shape[1]
        dB = np.sum(dZ, axis=1, keepdims=True) #/ dZ.shape[1]

        # calculate momentum gradient as well as weighted average
        self.vdW = self.vdW * self.beta1 + (1 - self.beta1) * dW
        self.vdB = self.vdB * self.beta1 + (1 - self.beta1) * dB
        self.sdW = self.beta2 * self.sdW + (1 - self.beta2) * np.square(dW)
        self.sdB = self.beta2 * self.sdB + (1 - self.beta2) * np.square(dB)

        # bias correct values
        vdWcor = self.vdW / (1 - self.beta1 ** self.iteration)
        vdBcor = self.vdB / (1 - self.beta1 ** self.iteration)
        sdWcor = self.sdW / (1 - self.beta2 ** self.iteration)
        sdBcor = self.sdB / (1 - self.beta2 ** self.iteration)

        # update W and B
        nW = W - self.lr * vdWcor / (np.sqrt(sdWcor) + self.epsilon)
        nB = B - self.lr * vdBcor / (np.sqrt(sdBcor) + self.epsilon)
        return nW, nB
