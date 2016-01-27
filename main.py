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


def find_nearest_neighbor(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor([1,1],[[1,2], [2,2], [0,0]])
    [1, 2]
    """
    nearest_neighbor = (0, np.inf)
    for n in neighborhood:
        n_dist = (n, neighbor_distance(neighbor, n))
        if n_dist < nearest_neighbor[1]:
            nearest_neighbor = (n, n_dist)
    return neighborhood[nearest_neighbor[0]]

def find_nearest_neighbor_index(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor_index([1,1],[[1,2], [2,2], [0,0]])
    0
    """
    f = np.vectorize(neighbor_distance, excluded='x2')
    return np.argmin(f(neighborhood, neighbor))

def find_nearest_neighbor_fast(neighbor, neighborhood):
    """ Calculates the nearest neighbor for a given array
    >>> find_nearest_neighbor_fast([1,1],[[1,2], [2,2], [0,0]])
    [1, 2]
    """
    f = np.vectorize(neighbor_distance, excluded='x2')
    return neighborhood[np.argmin(f(neighborhood, neighbor))]

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
    bottle_neck = 16
    encoder_dim = 250
    decoder_dim = 200
    batch_size = 128
    nb_epoch = 5
    activation_fnc = 'relu'

    # the data, shuffled and split between train and test sets
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[0:1000, :]
    x_test = x_test[0:1000, :]

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

    # validation with nearest neighbor
    #    compare_autoencoder_outputs(x_test, model, indices=[1, 2, 3, 4])

    encoder = Sequential()
    for i, layer in enumerate(model.layers):
        if i == 2:
            break
        encoder.add(layer)

    encoder.compile(loss='mean_squared_error',
                  optimizer=RMSprop())
# °;ö;°    # sanity checking weights
# °;ö;°    print(all((model.layers[0].get_weights()[1] == encoder.layers[0].get_weights()[1])))
# °;ö;°    print(all((model.layers[1].get_weights()[1] == encoder.layers[1].get_weights()[1])))
# °;ö;°
# °;ö;°    neighborhood = encoder.predict(x_train)
# °;ö;°    neighbor = encoder.predict(x_train[1:2,:])
# °;ö;°
# °;ö;°    # sanity checking predict
# °;ö;°    plt.matshow(neighborhood[1].reshape((4,4)))
# °;ö;°    plt.matshow(neighbor.reshape((4,4)))
# °;ö;°
# °;ö;°    i = find_nearest_neighbor_index(neighbor, neighborhood)
# °;ö;°    plt.matshow(neighbor.reshape(4,4))
# °;ö;°    plt.matshow(neighborhood[1].reshape(4,4))
# °;ö;°    plt.matshow(x_train[i].reshape((28,28)))
# °;ö;°    plt.matshow(x_train[1].reshape((28,28)))
# °;ö;°    plt.show()

def compare_autoencoder_outputs(imgs, model, indices=[0], img_dim=(28, 28)):
    pred = model.predict(imgs)
    for i in indices:
        tup = (imgs[i].reshape(img_dim), pred[i].reshape(img_dim))
        plt.matshow(tup[0])
        plt.matshow(tup[1])
    plt.show()


if __name__ == "__main__":
    run_deep_autoencoder()
