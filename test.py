#!/usr/bin/env python3

import numpy as np
from layers.dense import Dense
from layers.activations import Tanh


# (4,2,1)

#X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))

# (2,4)
X = np.reshape([[0,0,1,1],
                [0,1,0,1]], (2,4))

X = np.random.randn(784,10000)

network = [
    Dense(784,56,[]),
    Tanh(),
    Dense(56,10,[]),
    Tanh()
]

output = X
for layer in network:
    output = layer.forward(output)

grad = output
for layer in reversed(network):
    grad = layer.backward(grad)

print(output.shape)
