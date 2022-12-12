import numpy as np
import tensorflow as tf


## Run functions eagerly to allow numpy conversions.
## Enable experimental debug mode to suppress warning (feel free to remove second line)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


###############################################################################################


def get_data():
    """
    Loads CIFAR10 training and testing datasets

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
            D0: TF Dataset training subset
            D1: TF Dataset testing subset
        D_info: TF Dataset metadata
    """

    import tensorflow_datasets as tfds

 
    (D0, D1), D_info = tfds.load(
        "cifar10", as_supervised=True, split=["train", "test"], with_info=True
    )

    X0, X1 = [np.array([r[0] for r in tfds.as_numpy(D)]) for D in (D0, D1)]
    Y0, Y1 = [np.array([r[1] for r in tfds.as_numpy(D)]) for D in (D0, D1)]

    return X0, Y0, X1, Y1, D0, D1, D_info


###############################################################################################


def run_task(data, task, subtask="all", epochs=None, batch_size=None):
    """
    Runs model on a given dataset.

    :param data: Input dataset to train on
    :param task: 1 => train the model with tf.keras.layers (tf.keras.layers.Conv2D,
                      tf.keras.layers.Dense, etc.). Only use this to find a good model. This
                      will NOT be enabled on the autograder.
                 2 => train the model with some/all of the layers
                      you've implemented in layers_keras.py (which layers to substitute are
                      specified in subtask)
                 3 => train the model with your manual implementation of Conv2D
    :param subtask: 1 => train the model with the Conv2D layer you've implemented in layers_keras.py
                    2 => train the model with the BatchNormalization layer you've implemented in
                         layers_keras.py
                    3 => train the model with the Dropout layer you've implemented in
                         layers_keras.py
                    all => use all layers from layers_keras.py

    :return trained model
    """
    import conv_model_shared     ## Where your model, preprocessing, and augmentation pipelines are.


    ## Retrieve data from tuple
    X0, Y0, X1, Y1, D0, D1, D_info = data

 

    



    ## Retrieve the actual CNN model given which version of convolution,
    ## batch normalization, and dropout we're using
    args = conv_model_shared.get_default_CNN_model(
       
    )

    ## Prioritize function arguments
    if task != 3:
        if epochs is None:
            epochs = args.epochs
        if batch_size is None:
            batch_size = args.batch_size
        X0_sub, Y0_sub = X0, Y0
        X1_sub, Y1_sub = X1, Y1
    else:
        # If task 3 (using manual Conv2D implemenatation), make dataset/training
        # extremely small (it will be slow)
        if epochs is None:
            epochs = 2
        if batch_size is None:
            batch_size = 250
        X0_sub, Y0_sub = X0[:250], Y0[:250]
        X1_sub, Y1_sub = X1[:250], Y1[:250]

    # Training model
    print("Starting Model Training")
    history = args.model.fit(
        X0_sub, Y0_sub,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_data = (X1_sub, Y1_sub),
    )

    return args.model


