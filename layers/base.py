#!/usr/bin/env python3

from abc import abstractmethod, ABC

class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
