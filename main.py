#!/usr/bin/env python3

from layers.activations import Tanh
from util.loss import mse, mse_prime
from layers.denseLayer import DenseLayer
import numpy as np

# (4,2) -> (4,2,1)
X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
# (4,1) -> (4,1,1)
Y = np.reshape([[0], [1], [1], [0]], (4,1,1))

network = [
    DenseLayer(2,3),
    Tanh(),
    DenseLayer(3,1),
    Tanh()
]

# hyperparameters
epochs = 10000
learning_rate = 0.1

def test(network, X,Y):
    for x, y in zip(X,Y):
        output = predict(network, x)
        print(output, y[0][0])


# training loop

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return 1 if output[0][0] >= 0.95 else 0

for e in range(epochs+1):
    error = 0
    #print(predict(network, X))
    for x, y in zip(X,Y):
        # forward step
        output = predict(network, x)

        # compute error
        error += mse(y, output)

        # backward step
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    if e % 1000 == 0:
        error /= len(X)
        test(network, X, Y)
        print('%d/%d, error=%f, accuracy=%f' % (e, epochs, error, 0))
test(network, X,Y)
