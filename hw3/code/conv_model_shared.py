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

    #Augment
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

    #Initialize shared basis layers orthogonally
    initializer = tf.keras.initializers.Orthogonal()

    model = Shared_ResNet(
            [tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            #shared basis 1 [3]
            tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer),
        
            tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            #shared basis 2 [9]
            tf.keras.layers.Conv2D(2, 3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer),
        
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            #shared basis 3 [15]
            tf.keras.layers.Conv2D(4, 3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer),
        
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

  
    

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr = 0.1,momentum=0.9,decay=4e-5),
        loss="categorical_crossentropy",  
        metrics=["categorical_accuracy"],
    )

    ## TODO 4: Pick an appropriate number of epochs and batch size to use for training
    ## your model. Note that the autograder will time out after 10 minutes.
    return SimpleNamespace(model=model, epochs=30, batch_size=300)


###############################################################################################


class Shared_ResNet(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.


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

        


    def batch_step_shared(self, data, training=False):

        x_raw, y_raw = data

        x = self.input_prep_fn(x_raw)
        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            
            self.shared_basis_1 = self._self_tracked_trackables[3]._trainable_weights[0]
            self.shared_basis_2 = self._self_tracked_trackables[9]._trainable_weights[0]
            self.shared_basis_3 = self._self_tracked_trackables[15]._trainable_weights[0]

            #orthoganal regularization term added to the loss function
            val = self.avg_sim(self.shared_basis_1) + self.avg_sim(self.shared_basis_2) + self.avg_sim(self.shared_basis_3)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) + val

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

         
        else:
            #loss
            #print(self.metrics[0].result().numpy())

            #accuracy
            #print(self.metrics[1].result().numpy())


            str = 'shared_metrics.csv'
            with open(str,'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.metrics[0].result().numpy(),self.metrics[1].result().numpy()])
                

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

 
        return {m.name: m.result() for m in self.metrics}

    def avg_sim(self,shared_basis):
        #shared basis is an input tensor array
        cnt_sim = 0
        sim = 0


        num_all_basis = shared_basis.shape[2]
        all_basis = shared_basis

     

        B = tf.concat(all_basis,0)
      
        B = tf.reshape(B,(num_all_basis, -1))
      

            # compute orthogonalities btwn all baisis 
     
        D = tf.linalg.matmul(B, tf.transpose(B)) 

            # make diagonal zeros
        D = (D - tf.eye(num_all_basis, num_all_basis))**2

        sim += tf.math.reduce_sum(D[0:num_all_basis,0:num_all_basis])
        cnt_sim += num_all_basis**2

        avg_sim = sim / cnt_sim

        return avg_sim
            


    def train_step(self, data):
        return self.batch_step_shared(data, training=True)

    def test_step(self, data):
        return self.batch_step_shared(data, training=False)

    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)
