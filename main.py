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
from keras.optimizers import Adam
from keras.layers.core import Dense

from nearestneighbor import find_nearest_neighbor_index

import cPickle



def run_deep_autoencoder(dataset, img_dim=20**2, img_shape=(20,20), bottle_neck=16, classes={1: "happy", 2: "sad", 3 : "frustrated", 4 : "winking"}):
    """
    funcion docsting
    """
    np.random.seed(1337)  # for reproducibility
    encoder_dim = 250
    decoder_dim = 250
    batch_size = 128
    nb_epoch = 100
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = dataset

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

    # validation with nearest neighbor
    #    compare_autoencoder_outputs(x_test, model, indices=[1, 2, 3, 4])

    encoder = Sequential()
    for i, layer in enumerate(model.layers):
        if i == 2:
            break
        encoder.add(layer)

    encoder.compile(loss='mean_squared_error', optimizer=RMSprop())
    # sanity checking weights
    print('Checking if the encoders weights equals\
            the beginning of the autoencoder')
    print(all((model.layers[0].get_weights()[1] == encoder.layers[0].get_weights()[1])))
    print(all((model.layers[1].get_weights()[1] == encoder.layers[1].get_weights()[1])))

    neighborhood = encoder.predict(x_train)
    #neighbor = encoder.predict(x_train[1:2,:])

    correct = 0.
    for testindex in range(len(x_test)):
        neighbor = encoder.predict(x_test[testindex: testindex+1,:])
        i = find_nearest_neighbor_index(neighbor, neighborhood)
        if y_train[i][0] == y_test[testindex][0]:
            correct += 1.
    precision = correct / len(x_test)
    print("Precision: " + str(round(precision, 3)))


def compare_autoencoder_outputs(imgs, model, indices=[0], img_dim=(28, 28)):
    pred = model.predict(imgs)
    for i in indices:
        tup = (imgs[i].reshape(img_dim), pred[i].reshape(img_dim))
        plt.matshow(tup[0])
        plt.matshow(tup[1])
    plt.show()


def load_smileys_dataset(filename="./smiley.pkl", n_train=900, n_test=99,img_dim=20*20):
    """ load smiley database.
    :returns: (x_train, y_train) , (x_test, y_test)
    """
    images, labels = cPickle.load(open(filename, "rb"))

    x_train = images[0:n_train, :]
    y_train = labels[0:n_train, :]
    x_test = images[n_train +1:n_train +1 + n_test, :]
    y_test = labels[n_train +1:n_train +1 + n_test, :]
    x_train = x_train.reshape(-1, img_dim)
    x_test = x_test.reshape(-1, img_dim)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(x_train.shape, 'train sample shape')
    print(x_test.shape, 'test sample shape')
    print(y_train.shape, 'train y sample shape')
    print(y_test.shape, 'test y sample shape')

    return (x_train, y_train) , (x_test, y_test)


if __name__ == "__main__":
    data = load_smileys_dataset()
    run_deep_autoencoder(dataset=data)
    #load_smileys()
