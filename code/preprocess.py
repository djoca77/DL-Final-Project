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

    ## https://www.cs.toronto.edu/~kriz/cifar.html
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    data_files = {
        "train": [f"data_batch_{i+1}" for i in range(5)],
        "test": ["test_batch"],
    }[subset]
    data_meta = f"{data_path}/cifar/batches.meta"
    data_files = [f"{data_path}/cifar/{file}" for file in data_files]

    if subset == "train": #which value to use for reshaping, important later
        n = 50000
    else:
        n = 10000

    cifar_dict = {  ## HINT: Might help to start out with this
        b"data": [],
        b"labels": [],
    }

    #go through each of the batch files unpickle them, and combine them together into a single array
    for i in range(0, len(data_files)): 
        with open(data_files[i], 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            cifar_dict[b"data"].extend(dict[b"data"])
            cifar_dict[b"labels"].extend(dict[b"labels"])

    with open(data_meta, 'rb') as fo:
        cifar_meta = pickle.load(fo, encoding='bytes')

    image = cifar_dict[b"data"]
    label_num = cifar_dict[b"labels"]
    label_names = cifar_meta[b"label_names"]

    label = []

    #get the descriptive names for the labels and append them to the list, this is done by indexing
    #into the numerical labels gotten from the cifar_dict to get the number, and indexing into label names 
    #to get the corresponding name to add to the list
    for i in range(0, len(label_num)):
        label.append(label_names[label_num[i]])

    #change everything to numpy arrays, and reshape and transpose image so that we have the correct dimensions
    #in the correct places
    label = np.array(label)
    image = np.reshape(image, (n, 3, 32, 32))
    image = np.transpose(image, (0,2,3,1))
    image = np.array(image, dtype=np.uint8)
    label_names = np.array(label_names)

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label)
    #print(int(n/batch_size))

    #image = tf.even_splits(image, int(n/batch_size))

    return image, label


def shuffle_data(image_full, label_full, seed):
    """
    Shuffles the full dataset with the given random seed.

    NOTE: DO NOT EDIT

    It's important that you don't edit this function,
    so that the autograder won't be confused.

    :param: the dataset before shuffling
    :return: the dataset after shuffling
    """
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full


def get_specific_class(image_full, label_full, specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a specific digit.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image_full: the image array returned by the get_data function
    :param label_full: the label array returned by the get_data function
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """
    
    label = label_full[label_full == specific_class] #mask to get only what we want and nothign else
    image = image_full[label_full == specific_class]
    label = label[:num] #cut off after certain point specified by num
    image = image[:num]

    return image, label


def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a list of specific digits.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image: the image array returned by the get_data function
    :param label: the label array returned by the get_data function
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """

    image = []
    label = []

    #go through each of the classes, get the specific class' images and labels up to num, and add them to 
    #an array, making sure at the end to make it a numpy array
    for i in class_list: 
        image_list, label_list = get_specific_class(image_full, label_full, specific_class=i, num=num)
        image.extend(image_list)
        label.extend(label_list)

    image = np.array(image, dtype=image_list.dtype)
    label = np.array(label, dtype=label_list.dtype)

    return image, label