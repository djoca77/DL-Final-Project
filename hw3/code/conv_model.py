from types import SimpleNamespace

import numpy as np
import tensorflow as tf



import csv


###############################################################################################


def get_default_CNN_model(

):
    """
    Sets up your model architecture and compiles it using the appropriate optimizer, loss, and
    metrics.

    :returns compiled model
    """

    #Augment input
    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )

    augment_fn = tf.keras.Sequential([

            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation((-0.001,0.001)),
        ])

    model = ResNet(
            [tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            
            #Basic blocks
            tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        
            #Basic blocks
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        
            #Basic blocks
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
    
        
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10,activation='sigmoid')

            ],
            input_prep_fn=input_prep_fn,
            output_prep_fn=output_prep_fn,
            augment_fn=augment_fn,
      
        )

    #Compile your model with SGD
    

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr = 0.1,momentum=0.9,decay=4e-5),
        loss="categorical_crossentropy",  
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=30, batch_size=300)


###############################################################################################


class ResNet(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.

    DO NOT EDIT

    :param input_prep_fn: Modifies input images prior to running the forward pass
    :param output_prep_fn: Modifies input labels prior to running forward pass
    :param augment_fn: Augments input images prior to running forward pass
    """

    def __init__(
        self,
        *args,
        input_prep_fn=lambda x: x,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs

        
    ):
        super().__init__(*args, **kwargs)
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

        

    def batch_step(self, data, training=False):

        x_raw, y_raw = data

        x = self.input_prep_fn(x_raw)
        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            # Compute the loss value (the loss function is configured in `compile()`)
            
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            str = 'metrics.csv'
            with open(str,'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.metrics[0].result().numpy(),self.metrics[1].result().numpy()])

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}



    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)
