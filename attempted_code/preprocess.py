import gzip
import pickle

import numpy as np
import tensorflow as tf

def get_data_CIFAR(subset, batch_size, data_path="../data"):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ...,
    as well as test_batch, so you'll need to combine all train batches
    into one batch. Each of these files is a Python "pickled"
    object produced with cPickle. The code below will open up each
    "pickled" object (i.e. each file) and return a dictionary.

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the CIFAR data
    :return:
        inputs (NumPy array of uint8),
        labels (NumPy array of string),
        label_names (NumPy array of strings)
    """

    Cifar10=tf.keras.datasets.cifar10 # Loading the dataset

    (xtrain,ytrain),(xtest,ytest)= Cifar10.load_data()
    
    image = None
    label = None
    n = 0

    if subset == "train": #which value to use for reshaping, important later
        n = 50000
        image = xtrain
        label = ytrain
    else:
        n = 10000
        image = xtest
        label = ytest

    image = tf.convert_to_tensor(image, tf.float32)
    label = tf.one_hot(label, 10, dtype=tf.int32)
    label = tf.reshape(label, [n, 10])

    image = tf.split(image, int(n/batch_size))
    label = tf.split(label, int(n/batch_size))

    return image, label

