#!/usr/bin/env python3

import numpy as np
from .optimizer import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def optimize(self, output_gradient, input, weights, bias):
        weights_gradient = np.dot(output_gradient, input.T)
        weights -= self.lr * weights_gradient
        bias -= self.lr * output_gradient
        return weights, bias
