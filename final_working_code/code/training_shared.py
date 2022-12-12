import matplotlib.pyplot as plt
import tensorflow as tf
import   assignment1, conv_model_shared

data = assignment1.get_data()
X0, Y0, X1, Y1, D0, D1, D_info = data

D_info


D_info.features['label']._int2str



## You can use any list of 10 indices
sample_image_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sample_images = tf.cast(tf.gather(X0, sample_image_indices), tf.float32)
sample_labels = tf.gather(Y0, sample_image_indices)

fig, ax = plt.subplots(3, 10)
fig.set_size_inches(24, 8)

#Get model
args = conv_model_shared.get_default_CNN_model()

preprocessed_images = args.model.input_prep_fn(sample_images)
augmented_images = args.model.augment_fn(preprocessed_images)


#Run model
cnn_model = assignment1.run_task(data, 1, epochs=30, batch_size=300)

