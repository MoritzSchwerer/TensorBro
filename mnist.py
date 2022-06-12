#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from layers.denseLayer import DenseLayer
from layers.dense import Dense
from layers.activations import ReLu, Tanh
from util.loss import mse, mse_prime

from optimizers.gradientdescentMM import GradientDescentMM

def preprocess_data(x,y,limit):
    # reshape and normalisation
    x = x.reshape(x.shape[0], 28 * 28).T
    x = x.astype("float32") / 255

    # vactor encode result
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10).T
    return x[:,:limit], y[:,:limit]



# load Mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test   = preprocess_data(x_test, y_test, 100)

num_batches = 100
x_train = np.split(x_train, num_batches, axis=1)
y_train = np.split(y_train, num_batches, axis=1)

# hyperparameters
epochs = 1000
learning_rate = 0.1

optimizer1 = GradientDescentMM(28*28, 28*2)
optimizer2 = GradientDescentMM(28*2, 10)
# nural network
network = [
    Dense(28*28, 28*2, optimizer1),
    Tanh(),
    Dense(28*2,10, optimizer2),
    Tanh()
]


def test(network, X,Y):
    output = predict(network, X)
    pred = np.argmax(output, axis=0)
    act = np.argmax(Y,axis=0)
    right = np.count_nonzero(pred==act)
    return right / act.shape[0]

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# training loop
for e in range(epochs+1):

    # batch
    for batch in range(num_batches):

        # forward step
        output = predict(network, x_train[batch])

        # compute error
        error = np.sum(mse(y_train[batch], output))
        #print("error shape: ", error.shape)

        # backward step
        grad = mse_prime(y_train[batch], output)
        for layer in reversed(network):
            grad = layer.backward(grad)

    if e % 10 == 0:
        error /= len(x_train[0])
        acc = test(network, x_test,y_test)
        print('%d/%d, error=%f, accuracy=%f' % (e, epochs, error, round(acc,2)))
