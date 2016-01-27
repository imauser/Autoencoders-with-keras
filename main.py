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


def run_deep_autoencoder():
    """
    funcion docsting
    """
    np.random.seed(1337)  # for reproducibility
    batch_size = 128
    nb_epoch = 100
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

bottle_neck = 500

activation_fnc = 'relu'

def compare_autoencoder_outputs(imgs, model, indices=[0], img_dim=(28,28)):
    pred = model.predict(imgs)
    for i in indices:
        tup = (imgs[i].reshape(img_dim),pred[i].reshape(img_dim))
        plt.matshow(tup[0])
        plt.matshow(tup[1])
    plt.show()


def run_deep_autoencoder():
    img_dim = 28*28
    bottle_neck = 100
    encoder_dim = 250
    decoder_dim = 200
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
    compare_autoencoder_outputs(X_test, model, indices=[1,2,3,4])

if __name__ == "__main__":
    run_deep_autoencoder()
