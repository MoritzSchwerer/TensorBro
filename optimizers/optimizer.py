#!/usr/bin/env python3

from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __init__(self, learning_rate):
        pass

    @abstractmethod
    def optimize(self, output_gradient, input, weights, bias):
        pass
