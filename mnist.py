#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from layers.denseLayer import DenseLayer
from layers.activations import Tanh
from util.loss import mse, mse_prime

def preprocess_data(x,y,limit):
    # reshape and normalisation
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255

    # vactor encode result
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]



# load Mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test   = preprocess_data(x_test, y_test, 100)

# nural network
network = [
    DenseLayer(28*28, 28*7),
    Tanh(),
    DenseLayer(28*7,10),
    Tanh()
]

# hyperparameters
epochs = 1000
learning_rate = 0.1

def test(network, X,Y):
    right = 0
    for x, y in zip(X,Y):
        output = predict(network, x)
        pred = np.argmax(output)
        act = np.argmax(y)
        #print(pred, act)
        right += 1 if pred == act else 0
    return right / len(X)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# training loop
i = 0
for e in range(epochs+1):
    error = 0
    i += 0.00009
    #print(predict(network, X))
    for x, y in zip(x_train,y_train):
        # forward step
        output = predict(network, x)

        # compute error
        error += mse(y, output)

        # backward step
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate - i)

    if e % 10 == 0:
        error /= len(x_train)
        acc = test(network, x_test,y_test)
        print('%d/%d, error=%f, accuracy=%f' % (e, epochs, error, round(acc,2)))
