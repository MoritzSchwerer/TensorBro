#!/usr/bin/env python3

from .base import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def forward(self, input):
        self.input = input
        return self.f(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.f_prime(self.input))
