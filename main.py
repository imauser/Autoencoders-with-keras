#!/usr/bin/python
"""
Module docsting
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense

import cPickle


def find_nearest_neighbor(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor([1,1],[[1,2], [2,2], [0,0]])
    [0, 0]
    """
    nearest_neighbor = (0, np.inf)
    for n in neighborhood:
        n_dist = (n, neighbor_distance(neighbor, n))
        if n_dist < nearest_neighbor[1]:
            nearest_neighbor = (n, n_dist)
    return nearest_neighbor[0]


def neighbor_distance(x1, x2):
    """ :return: distance of the two scalars or matrices in l2/frobenius norm
    """
    return np.linalg.norm(np.subtract(x1, x2))


def run_deep_autoencoder():
    """
    funcion docsting
    """
    np.random.seed(1337)  # for reproducibility
    img_dim = 28*28
    bottle_neck = 5
    encoder_dim = 250
    decoder_dim = 250
    batch_size = 128
    nb_epoch = 100
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # x_train = x_train[0:1000, :]
    # x_test = x_test[0:1000, :]

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # non-autoencoder

    model = Sequential()
    model.add(Dense(output_dim=encoder_dim, input_dim=img_dim,
                    activation=activation_fnc, init='uniform'))
    model.add(Dense(output_dim=bottle_neck, activation=activation_fnc,
                    init='uniform'))
    model.add(Dense(output_dim=decoder_dim, activation=activation_fnc,
                    init='uniform'))
    model.add(Dense(input_dim=decoder_dim, activation=activation_fnc,
                    output_dim=img_dim, init='uniform'))
    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())
    model.fit(x_train, x_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(x_test, x_test), show_accuracy=False)

    # validation mit nearest neighbor
    # bottleneck small
    compare_autoencoder_outputs(x_test, model, indices=[1, 2, 3, 4])


def compare_autoencoder_outputs(imgs, model, indices=[0], img_dim=(28, 28)):
    pred = model.predict(imgs)
    for i in indices:
        tup = (imgs[i].reshape(img_dim), pred[i].reshape(img_dim))
        plt.matshow(tup[0])
        plt.matshow(tup[1])
    plt.show()


def load_smileys():
    """load smileys 

    :arg1: TODO
    :returns: TODO

    """
    images, lables = cPickle.load(open("./smiley.pkl", "rb"))
    print("images")
    print(len(images))

    np.random.seed(1337)  # for reproducibility
    img_dim = 20*20
    bottle_neck = 250
    encoder_dim = 250
    decoder_dim = 250
    batch_size = 128
    nb_epoch = 500
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    x_train = images[0:800, :]
    x_test = images[801:999, :]
    x_train = x_train.reshape(-1, 400)
    x_test = x_test.reshape(-1, 400)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # x_train = x_train[0:100, :]
    # x_test = x_test[0:100, :]

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # non-autoencoder

    model = Sequential()
    model.add(Dense(output_dim=encoder_dim, input_dim=img_dim,
                    activation=activation_fnc, init='uniform'))
    model.add(Dense(output_dim=bottle_neck, activation=activation_fnc,
                    init='uniform'))
    model.add(Dense(output_dim=decoder_dim, activation=activation_fnc,
                    init='uniform'))
    model.add(Dense(input_dim=decoder_dim, activation=activation_fnc,
                    output_dim=img_dim, init='uniform'))
    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())
    model.fit(x_train, x_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(x_test, x_test), show_accuracy=False)

    # validation mit nearest neighbor
    # bottleneck small
    compare_autoencoder_outputs(x_test, model, indices=[1, 2, 3, 4], img_dim=(20, 20))

if __name__ == "__main__":
    # run_deep_autoencoder()
    load_smileys()
