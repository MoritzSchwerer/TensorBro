#!/usr/bin/env python3

from .activation import Activation
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
