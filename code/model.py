import tensorflow as tf
import copy


class LambdaLayer(tf.keras.layers.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def call(self, x):
        return self.lambd(x)

class BasicBlock_NonShared(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, stride=1, option='A'):
        super(BasicBlock_NonShared, self).__init__()
        
        self.unique_rank = unique_rank
        
        self.total_rank_1 = unique_rank
        self.total_rank_2 = unique_rank
        
        self.basis_conv1 = tf.keras.layers.Conv2D(unique_rank, 3, strides=stride, padding='same', use_bias=False)
        self.basis_bn1 = tf.keras.layers.BatchNormalization()
        self.coeff_conv1 = tf.keras.layers.Conv2D(planes, 1, strides=1, padding='same', use_bias=False)
        
        #self.bn1 = tf.keras.layers.BatchNormalization(planes)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.basis_conv2 = tf.keras.layers.Conv2D(unique_rank, 3, strides=1, padding='same', use_bias=False)
        self.basis_bn2 = tf.keras.layers.BatchNormalization()
        self.coeff_conv2 = tf.keras.layers.Conv2D(planes, 1, strides=1, padding='same', use_bias=False)
        
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            tf.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = tf.keras.layers.Sequential(
                     tf.keras.layers.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     tf.keras.layers.BatchNormalization()
                )

    def call(self, x):           
        out = self.basis_conv1(x)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = tf.nn.relu(out)
        
        out = self.basis_conv2(out)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        #out += self.shortcut(x)
        out = tf.nn.relu(out)
        
        return out


# Original BasicBlock
class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(planes, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            tf.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

                print(1111)
            elif option == 'B':
                self.shortcut = tf.keras.Sequential(
                     tf.keras.layers.Conv2D(self.expansion * planes, 1, strides=stride, use_bias=False),
                     tf.keras.layers.BatchNormalization()
                )

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #out += self.shortcut(x)
        out = tf.nn.relu(out)
        
        return out



# ResNet for unique basis only models
class ResNet_NonShared(tf.keras.layers.Layer):
    def __init__(self, block_basis, block_original, num_blocks, unique_rank, num_classes=10):
        super(ResNet_NonShared, self).__init__()
        self.in_planes = 16

        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank*1, stride=1)
        
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, stride=2)
        
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, stride=2)
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D() #AdaptiveAvgPool2d((1, 1))
        self.fc = tf.keras.layers.Dense(num_classes)
        
        for m in self.get_config():
            if isinstance(m, tf.keras.layers.Conv2D):
                #initialize every con2d first, then initialize shared basis again later
                tf.keras.initializers.HeNormal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)): #nn.GroupNorm)
                tf.keras.initializers.Constant(m.weight, 1)
                tf.keras.initializers.Constant(m.bias, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "arg1": self.in_planes,
            "arg2": self.conv1,
            "arg3": self.bn1,
            "arg4": self.layer1,
            "arg5": self.layer2,
            "arg6": self.layer3,
            "arg7": self.avgpool,
            "arg8": self.fc
        })
        return config

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank))

        return tf.keras.Sequential(layers = layers)
    
    def _make_layer_original(self, block_original, planes, blocks, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_original(self.in_planes, planes, stride))

        return tf.keras.Sequential(layers = layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = tf.keras.layers.Flatten(x)
        x = self.fc(x)
     
        return x


# A model without shared bases in each residual block group. Only unique bases used are in each block.
def ResNet56_NonShared(unique_rank):
    return ResNet_NonShared(BasicBlock_NonShared, BasicBlock, [9, 9, 9], unique_rank)