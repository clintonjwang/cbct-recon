__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2017, WGAN with Keras"
__email__ = "xtong@andrew.cmu.edu"

from keras.layers.convolutional import Conv2D, Conv2DTranspose
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ELU
from keras.layers.core import Dense, Flatten, Reshape, Activation
from keras.layers import Input
from keras.initializers import RandomNormal
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model

class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        #self.name = 'mnist/dcgan/discriminator'
        self.initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        self.regularizer = regularizers.l2(2.5e-5)

    def __call__(self):
        model = Sequential()
        model.add(Reshape((28, 28, 1), input_shape=(784,)))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), \
            kernel_initializer=self.initializer))
        model.add(ELU())

        model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), \
            kernel_initializer=self.initializer))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(1024, kernel_initializer=self.initializer))
        model.add(ELU())

        model.add(BatchNormalization())

        model.add(Dense(2, activation='softmax'))

        return model


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        #self.name = 'mnist/dcgan/generator'
        self.initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        self.regularizer = regularizers.l2(2.5e-5)

    def __call__(self):
        model = Sequential()

        model.add(Dense(1024, kernel_initializer=self.initializer, \
            kernel_regularizer=self.regularizer, input_shape=(self.z_dim,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(7 * 7 * 128, kernel_initializer=self.initializer, \
            kernel_regularizer=self.regularizer))
        model.add(Reshape((7, 7, 128)))

        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same',\
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same',\
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
        model.add(Activation('sigmoid'))
        model.add(Reshape((784,)))

        return model