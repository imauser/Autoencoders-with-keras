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



def run_deep_autoencoder():
    """
    funcion docsting
    """
    np.random.seed(1337)  # for reproducibility
    img_dim = 28*28
    bottle_neck = 16
    encoder_dim = 250
    decoder_dim = 250
    batch_size = 128
    nb_epoch = 10
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[0:1000, :]
    x_test = x_test[0:1000, :]

    print(x_train.shape, 'train sample shape')
    print(x_test.shape, 'test sample shape')
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
    testindex = 20
    neighbor = encoder.predict(x_test[testindex: testindex+1,:])

    # neighbor shapes
    print(str(neighborhood.shape))
    print(str(neighbor.shape))

    i = find_nearest_neighbor_index(neighbor, neighborhood)
    print('index:' + str(i))
    plt.matshow(neighborhood[i].reshape(4,4))
    plt.matshow(neighbor.reshape(4,4))
    plt.matshow(x_train[i].reshape((28,28)))
    plt.matshow(x_test[testindex].reshape((28,28)))
    plt.show()

def compare_autoencoder_outputs(imgs, model, indices=[0], img_dim=(28, 28)):
    pred = model.predict(imgs)
    for i in indices:
        tup = (imgs[i].reshape(img_dim), pred[i].reshape(img_dim))
        plt.matshow(tup[0])
        plt.matshow(tup[1])
    plt.show()


def load_smileys_dataset(n_train=900, n_test=99,img_dim=20*20):
    images, labels = cPickle.load(open("./smiley.pkl", "rb"))
    x_train = images[0:n_train, :]
    y_train = labels[0:n_train, :]
    x_test = images[n_train +1:n_train +1 + n_test, :]
    x_test = labels[n_train +1:n_train +1 + n_test, :]
    x_train = x_train.reshape(-1, img_dim)
    x_test = x_test.reshape(-1, img_dim)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train) , (x_test, y_test)

def load_smileys():
    """load smileys

    :arg1: TODO
    :returns: TODO

    """
    images, lables = cPickle.load(open("./smiley.pkl", "rb"))
    print(str(lables.shape))
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
    x_train = images[0:900, :]
    x_test = images[901:999, :]
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
                  optimizer=Adam())
                  #optimizer=Adam(lr=0.01))
    model.fit(x_train, x_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=(x_test, x_test), show_accuracy=False)

    # compare_autoencoder_outputs(x_test, model, indices=[1, 2, 3, 4], img_dim=(20, 20))

if __name__ == "__main__":
    #run_deep_autoencoder()
    load_smileys()
