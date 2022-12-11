import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            ph = (fh - 1) // 2
            pw = (fw - 1) // 2
        elif self.padding == "VALID":
            ph, pw = 0, 0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        

        ## TODO: Convolve filter from above with the inputs.
        ## Note: Depending on whether you used SAME or VALID padding,
        ## the input and output sizes may not be the same

        if (w_in % sw == 0):
            pad_width = max(fw - sw, 0)
        else:
            pad_width = max(fw - (w_in % sw), 0)
        if (h_in % sh == 0):
            pad_height = max(fh - sh, 0)
        else:
            pad_height = max(fh - (h_in % sh), 0)


        p_left = pad_width // 2
        p_right = pad_width - p_left

        p_top = pad_height // 2
        p_bottom = pad_height - p_top



        ## Pad input if necessary
        padding = [[0,0],[p_top,p_bottom],[p_left,p_right],[0,0]]
        

        input1 = tf.pad(inputs,padding)



        ## Calculate correct output dimensions
        h_out = 1 + ((h_in - fh + 2*ph)//sh)
        w_out = 1 + ((w_in - fw + 2*pw)//sw)

        ## Iterate and apply convolution operator to each image

        out = np.zeros((bn,h_out,w_out,c_out))

        for i in range(bn):
            for j in range(c_out):
                for k in range(h_out):
                    for l in range(w_out):

                        in_temp = input1[i,k*sh:k*sh+fh,l*sw:l*sw+fw,:]
                        
                        out[i,k,l,j] = np.sum(np.multiply(self.kernel[:,:,:,j],in_temp))

        out = tf.convert_to_tensor(out, dtype=tf.float32)


        ## PLEASE RETURN A TENSOR using tf.convert_to_tensor(your_array, dtype=tf.float32)
        return out
