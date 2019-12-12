#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adithya
"""

import numpy as np
import scipy.io.wavfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, \
    Conv2D, MaxPooling2D, Flatten
import os.path
import datetime

fft_size = 512
hopsamp = fft_size//8


train_source_folder = os.path.join(
    '/home/adithya/Documents/IIST/AI_ML/Club_stuff/Spoken_Numbers', 'data', 'train')
test_source_folder = os.path.join(
    '/home/adithya/Documents/IIST/AI_ML/Club_stuff/Spoken_Numbers', 'data', 'test')


def get_signal(in_file, expected_fs=8000):
    """Load a wav file.
    If the file contains more than one channel, return a mono file by taking
    the mean of all channels.
    If the sample rate differs from the expected sample rate (default is 44100 Hz),
    raise an exception.
    Args:
        in_file: The input wav file, which should have a sample rate of `expected_fs`.
        expected_fs (int): The expected sample rate of the input wav file.
    Returns:
        The audio siganl as a 1-dim Numpy array. The values will be in the range [-1.0, 1.0]. fixme ( not yet)
    """
    fs, y = scipy.io.wavfile.read(in_file)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y*(1.0/32768)
    elif num_type == 'int32':
        y = y*(1.0/2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expected_fs:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)


def stft_for_reconstruction(x, fft_size=fft_size, hopsamp=hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.abs(np.fft.rfft(window*x[i:i+fft_size]))
                     for i in range(0, len(x)-fft_size, hopsamp)])


def Model1():
    model = Sequential()

    # A convolution layer helps encode data about a set of parameters:
    # (number of filters, dimension of filter, enable or disable
    # padding, dimensions of input matrix, activation function type)
    model.add(Conv2D(8, (3, 3), padding='valid',
                     input_shape=(122, 257, 1), activation='relu'))

    # A pooling layer reduces the size of the input matrix by
    # combining the data in the input matrix in groups of the filter
    # matrix and normalizing
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # We add more layers as per our neural network's architecture
    # For these convolution layers, we do not need to mention the
    # input dimension because it gets it from the previous layer
    model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # We now add a dropout layer, which randomly drops a fraction of the
    # input (it improves selection with time). We do this mainly because
    # some filters may have taken irrelevant details or there may be 2 or
    # more similar filter combination. We also want to reduce the size for
    # easier calculations.
    # Note: We could have added dropout layers between 2 Conv2D-MaxPooling
    # layer sets
    model.add(Dropout(0.5))

    # We now flatten the output to get a one-dimensional vector
    model.add(Flatten())

    # We now add some fully connected layers (Dense layers). The size
    # of the last dense layer must be the same as the length of the
    # output vector/number of output classes you have. To get a binary
    # type output (eg: 1000000000) we use a softmax layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


def one_hot_encoder(to_encode, unique_classes=[]):
    # Everything about this is explained below
    assert isinstance(to_encode, list)
    assert isinstance(unique_classes, list)

    if len(unique_classes) == 0:
        unique_classes = list(set(to_encode))

    encoded_array = np.zeros((len(to_encode), len(unique_classes)))

    for item, item_no in zip(to_encode, range(len(to_encode))):
        for check, i in zip(unique_classes, range(len(unique_classes))):
            if item == check:
                encoded_array[item_no, i] = 1

    return encoded_array


def data_import():
    testX = np.array([])
    trainX = np.array([])
    max_length = 0

    test_files = []
    train_files = []

    testY = []
    trainY = []

    # We get the names of the test and train files (ones ending with .wav)
    # and add them to the test list or train list. We also find the number
    # being said and use append that to the testY and trainY (the number in o/p)

    # Note: There are some assumptions here (based on dataset we have):
    #   1. the test files are all in one directory
    #   2. the train files are all in sub directories of another directory
    #   3. there are no other wav files in these folders
    #   4. the file name contains the number said as the first character

    for files in os.listdir(test_source_folder):
        if os.path.join(test_source_folder, files).endswith('.wav'):
            test_files.append(os.path.join(test_source_folder, files))
            testY.append(int(files[0]))

    for folder in os.listdir(train_source_folder):
        if os.path.isdir(os.path.join(train_source_folder, folder)):
            for files in os.listdir(os.path.join(train_source_folder, folder)):
                if os.path.join(train_source_folder, folder, files).endswith('.wav'):
                    train_files.append(os.path.join(
                        train_source_folder, folder, files))
                    trainY.append(int(files[0]))
    # print (train_files)

    # Since we are going to pad all our inputs, we need to find the one with
    # the largest size. Here we consider both the test and train files
    # Note: We need to pad our input to the same size while testing
    # unknown files (i.e. while using it).

    for file in test_files:
        stft_cnn = get_signal(file)
        if max_length < stft_cnn.size:
            max_length = stft_cnn.size

    for file in train_files:
        stft_cnn = get_signal(file)
        if max_length < stft_cnn.size:
            max_length = stft_cnn.size

    print(max_length)

    # We reshape our training and testing inputs to match the input for
    # the neural network (adding an axis in the end along which filters,
    # dimesnsions for the stft output etc)

    trainX = np.expand_dims(np.zeros(stft_for_reconstruction(
        np.zeros(max_length), fft_size, hopsamp).shape), axis=-1)
    testX = np.copy(trainX)

    trainX = np.broadcast_to(trainX, (len(trainY), ) + trainX.shape).copy()
    testX = np.broadcast_to(testX, (len(testY), ) + testX.shape).copy()

    test_count = 0

    # We then pad the input wave file and get the stft, which is
    # to be passed to the neural network. Note: The order in our
    # input and output files are same i.e. (trainX and trainY),
    # (testX and testY) have the same order of input

    for file in test_files:
        temp_padded_file = np.zeros(max_length)
        temp_file = get_signal(file)

        for i in range(len(temp_file)):
            temp_padded_file[i] = temp_file[i]

        temp_stft_file = stft_for_reconstruction(temp_padded_file)
        temp_stft_file = np.expand_dims(temp_stft_file, axis=-1)
        np.copyto(testX[test_count], temp_stft_file)
        test_count = test_count + 1

    train_count = 0

    for file in train_files:
        temp_padded_file = np.zeros(max_length)
        temp_file = get_signal(file)

        for i in range(len(temp_file)):
            temp_padded_file[i] = temp_file[i]

        temp_stft_file = stft_for_reconstruction(temp_padded_file)
        temp_stft_file = np.expand_dims(temp_stft_file, axis=-1)
        np.copyto(trainX[train_count], temp_stft_file)
        train_count = train_count + 1

    # Since we are going to use categorical_crossentropy for
    # our output, we need to one-hot-encode our input into the
    # classes present. One-Hot-Encoding means that there will be
    # an array of the length of the unique number of classes
    # and the class to which our output corresponds will have a
    # [1] value and others will have [0] value

    # Eg: Say we have an array [1, 2, 4, 6, 3, 5, 0]
    # Our unique classes are [0, 1, 2, 3, 4, 5, 6]
    # and the one-hot-encoding is
    # =>            1      : [0, 1, 0, 0, 0, 0, 0]
    # =>            2      : [0, 0, 1, 0, 0, 0, 0]
    # =>            4      : [0, 0, 0, 0, 1, 0, 0]
    # =>            6      : [0, 0, 0, 0, 0, 0, 1]
    # =>            3      : [0, 0, 0, 1, 0, 0, 0]
    # =>            5      : [0, 0, 0, 0, 0, 1, 0]
    # =>            0      : [1, 0, 0, 0, 0, 0, 0]

    trainY = one_hot_encoder(trainY, list(range(10)))
    testY = one_hot_encoder(testY, list(range(10)))

    return trainX, trainY, testX, testY


def train_model():
    # We now train the model, by getting the input files

    model = Model1()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    trainX, trainY, testX, testY = data_import()

    model.fit(trainX, trainY, epochs=20, shuffle=True)

    scores = model.evaluate(testX, testY)
    print('Accuracy: ', scores)

    return model, trainX, trainY, testX, testY


model, trainX, trainY, testX, testY = train_model()

if input('Do you want to save the model? (y/N)').strip().lower() == 'y':
    name = 'weights'+str(datetime.time)
    model.save_weights(name)

