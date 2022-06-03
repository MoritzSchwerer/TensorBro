#!/usr/bin/env python3

from .activation import Activation
from .base import Layer
import numpy as np

class ReLu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        deriv_relu = lambda x: x > 0
        super().__init__(relu, deriv_relu)


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        deriv_tanh = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, deriv_tanh)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(x))
        def deriv_sigmoid(x):
            pre = sigmoid(x)
            return pre * (1 - pre)
        super().__init__(sigmoid, deriv_sigmoid)


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        temp = np.exp(input)
        self.output = temp / np.sum(temp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
