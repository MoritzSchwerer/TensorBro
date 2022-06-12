#!/usr/bin/env python3

import numpy as np
from .optimizer import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def optimize(self, dZ, X, W, B):
        dW = np.matmul(dZ, X.T) #/ dZ.shape[1]
        dB = np.sum(dZ, axis=1, keepdims=True) #/ dZ.shape[1]
        nW = W - self.lr * dW
        nB = B - self.lr * dB
        return nW, nB
