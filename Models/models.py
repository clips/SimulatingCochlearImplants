import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import (Dense, BatchNormalization, Activation, Dropout,
                          Flatten,  Reshape)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True # don't pre-allocate entire GPU memory
config.allow_soft_placement=True     # fall back to CPU if no GPU op available
sess = tf.Session(config=config)
K.set_session(sess)


def perceptron(state, n_mels, inputlength, n_classes):

    # set random seed for replicability
    tf.set_random_seed(state['random_seed'])
    np.random.seed(state['random_seed'])
    init = glorot_uniform(seed=state['random_seed'])

    # initialize empty model
    model = Sequential()

    # flatten spectrograms for softmax layer
    model.add(Reshape(input_shape=(inputlength, n_mels),
                      target_shape=(inputlength * n_mels,)))

    # softmax layer
    model.add(Dense(units=n_classes, kernel_initializer=init))
    model.add(Activation('softmax'))

    print(model.summary())
    return model


def cnn(state, n_mels, inputlength, n_classes):

    # set random seed for replicability
    tf.set_random_seed(state['random_seed'])
    np.random.seed(state['random_seed'])
    init = glorot_uniform(seed=state['random_seed'])

    # initialize empty model
    model = Sequential()

    # add channel for convolutional layers
    model.add(Reshape(input_shape=(inputlength, n_mels),
                      target_shape=(inputlength, n_mels, 1),
                      name='reshape'))

    # convolutional layer 1
    model.add(Convolution2D(kernel_size=(state['kernel_size1'],
                                         state['kernel_size2']),
                            filters=state['number_filters'],
                            strides=(state['stride1'], state['stride2']),
                            padding="valid",
                            kernel_initializer=init,
                            input_shape=(inputlength, n_mels, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(state['conv_dropout'], seed=state['random_seed']))

    # convolutional layer 2
    model.add(Convolution2D(kernel_size=(state['kernel_size1'],
                                         state['kernel_size2']),
                            filters=state['number_filters'],
                            strides=(state['stride1'], state['stride2']),
                            padding="valid",
                            kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(state['conv_dropout'], seed=state['random_seed']))

    # convolutional layer 3
    model.add(Convolution2D(kernel_size=(state['kernel_size1'],
                                         state['kernel_size2']),
                            filters=state['number_filters'],
                            strides=(state['stride1'], state['stride2']),
                            padding="valid",
                            kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(state['conv_dropout'], seed=state['random_seed']))

    model.add(Flatten())

    # dense layer 1
    model.add(Dense(units=state['n_hidden_dense'],
                    kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(state['dense_dropout'], seed=state['random_seed']))

    # softmax layer
    model.add(Dense(units=n_classes, kernel_initializer=init))
    model.add(Activation('softmax'))

    print(model.summary())
    return model