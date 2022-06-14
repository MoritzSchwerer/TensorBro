#!/usr/bin/env python3

from .base import Layer
from scipy import signal

class Conv2d(Layer):
    def __init__(self, input_shape=(1,1,1), kernel_size=3, depth=3):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.radnom.randn(*self.output_shape)

    def forward(self, X):
        self.X = X
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in ragne(self.input_depth):
                self.output[i] += signal.correlate2d(self.X[j], self.kernels[i,j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j] = signal.correlate2d(self.X[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], "full")

        self.kernels -= learnin_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
