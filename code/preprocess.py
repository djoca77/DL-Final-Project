import gzip
import pickle

import numpy as np
import tensorflow as tf


def get_data_MNIST(subset, batch_size, data_path="../data"):
    """
    Takes in a subset of data ("train" or "test"), unzips the inputs and labels files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy array of labels).

    Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data.

    If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the MNIST data
    :return:
        inputs (NumPy array of float32)
        labels (NumPy array of uint8)
    """
    ## http://yann.lecun.com/exdb/mnist/
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    image = []
    label = []

    #for images
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream: 
        bytestream.read(16) #read past header
        image = np.frombuffer(bytestream.read(num_examples*784), dtype=np.uint8) #read the whole dataset from the buffer, 1D array
        image = np.divide(image, 255) #divide everything by 255
        image = np.reshape(image, (num_examples, 784)) #reshape it to the appropriate 2D array
    
    #for labels
    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8) #read past header
        label = np.frombuffer(bytestream.read(num_examples), dtype=np.uint8) #read everything from buffer, 1D array

    image = np.array(image, dtype=np.float32) #make numpy array of type float32
    label = np.array(label, dtype=np.uint8) #make numpy array of type int

    return image, label


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

