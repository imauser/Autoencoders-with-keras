from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 64
nb_epoch = 5

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#X_train = X_train[0:1000,:]
#X_test = X_test[0:1000,:]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

bottle_neck = 500

activation_fnc = 'relu'

def run_deep_autoencoder():
    img_dim = 28*28
    bottle_neck = 100
    bnEn = 250
    bnDe = 200
    # non-autoencoder
    model = Sequential()
    model.add(Dense(output_dim=bnEn, input_dim=img_dim, activation=activation_fnc, init='uniform'))
    model.add(Dense(output_dim=bottle_neck, input_dim=bnEn, activation=activation_fnc, init='uniform'))
    model.add(Dense(output_dim=bnDe, input_dim=bottle_neck, activation=activation_fnc, init='uniform'))
    model.add(Dense(input_dim=bnDe, activation=activation_fnc, output_dim=img_dim, init='uniform'))
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    np.random.seed(0)
    model.fit(X_train,X_train, nb_epoch=nb_epoch, batch_size=batch_size,
                validation_data=(X_test, X_test), show_accuracy=False)

    # validation mit nearest neighbor
    # bottleneck small

def run_non_ae():

    # non-autoencoder
    model = Sequential()
    model.add(Dense(output_dim=bottle_neck, input_dim=784, activation=activation_fnc, init='uniform'))
    model.add(Dense(input_dim=bottle_neck, activation=activation_fnc, output_dim=784, init='uniform'))
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    np.random.seed(0)
    model.fit(X_train,X_train, nb_epoch=nb_epoch, batch_size=batch_size,
                validation_data=(X_test, X_test), show_accuracy=False)
    y_image0 = model.predict(X_test)[0]
    x_image0 = X_test[0]

def run_ae():
    #creating the autoencoder
    ae = Sequential()

    encoder = containers.Sequential([Dense(output_dim=bottle_neck, input_dim=784, activation=activation_fnc, init='uniform')])
    decoder = containers.Sequential([Dense(input_dim=bottle_neck, activation=activation_fnc, output_dim=784, init='uniform')])


    #encoder = containers.Sequential([Dense(output_dim=600, input_dim=784, activation='relu'), Dense(output_dim=bottle_neck)])
    #decoder = containers.Sequential([Dense(output_dim=600, input_dim=bottle_neck, activation='relu'), Dense(output_dim=784)])
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

    ae.compile(loss='mean_squared_error', optimizer=RMSprop())

    ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=1, verbose=1, validation_data=[X_test, X_test], shuffle=True)

run_deep_autoencoder()
