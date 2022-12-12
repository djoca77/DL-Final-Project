import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import tensorflow_datasets as tfds


import   assignment, conv_model, layers_keras, layers_manual

data = assignment.get_data()
X0, Y0, X1, Y1, D0, D1, D_info = data


D_info


D_info.features['label']._int2str




import conv_model

## You can use any list of 10 indices
sample_image_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sample_images = tf.cast(tf.gather(X0, sample_image_indices), tf.float32)
sample_labels = tf.gather(Y0, sample_image_indices)

fig, ax = plt.subplots(3, 10)
fig.set_size_inches(24, 8)

args = conv_model.get_default_CNN_model()

preprocessed_images = args.model.input_prep_fn(sample_images)
augmented_images = args.model.augment_fn(preprocessed_images)



import assignment 

## You can test with more epochs later
cnn_model = assignment.run_task(data, 1, epochs=60, batch_size=300)

