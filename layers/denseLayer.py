#!/usr/bin/env python3

from .base import Layer
import numpy as np

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, optimizer):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.optimizer = optimizer

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        self.weights, self.bias = self.optimizer.optimize(output_gradient, self.input, self.weights, self.bias)
        return np.dot(self.weights.T, output_gradient)

    #def backward(self, output_gradient, learning_rate):
    #    dw = np.dot(output_gradient, self.input.T)
    #    self.sdw = self.beta * self.sdw + (1-self.beta) * dw
    #    db = output_gradient
    #    self.sdb = self.beta * self.sdb * (1-self.beta) * db
    #    self.weights -= learning_rate * self.sdw
    #    self.bias -= learning_rate * self.sdb
    #    return np.dot(self.weights.T, output_gradient)
