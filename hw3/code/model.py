import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import copy


class LambdaLayer(tf.keras.layers.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def call(self, x):
        return self.lambd(x)

# Original BasicBlock
class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, planes, stride=1, option='A'):
        
        self.conv1 = tf.keras.layers.Conv2D(planes, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1:
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


# BasicBlock for single basis models
class BasicBlock_SingleShared(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis, stride=1, option='A'):
        super(BasicBlock_SingleShared, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis = shared_basis
        
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        self.basis_conv1 = tf.keras.layers.Conv2d(unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = tf.keras.layers.BatchNormalization()
        self.coeff_conv1 = tf.keras.layers.Conv2d(planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.basis_conv2 = tf.keras.layers.Conv2d(unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = tf.keras.layers.BatchNormalization()
        self.coeff_conv2 = tf.keras.layers.Conv2d(planes, kernel_size=1, stride=1, padding=0, bias=False)
        
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
                self.shortcut = tf.keras.Sequential(
                     tf.keras.layers.Conv2D(self.expansion * planes, 1, strides=stride, use_bias=False),
                     tf.keras.layers.BatchNormalization()
                )

    def forward(self, x):           
        out = tf.concat((self.basis_conv1(x), self.shared_basis(x)),dim=1)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = tf.nn.relu(out)
        
        out = tf.concat((self.basis_conv2(out), self.shared_basis(out)),dim=1)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = tf.nn.relu(out)
        
        return out

class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):

        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
       #self.conv2 = tf.keras.layers.Conv2D(16, strides=1)
       # self.bn2 = tf.keras.layers.BatchNormalization()
       ## self.conv3 = tf.keras.layers.Conv2D(16, strides=1)
       # self.bn3 = tf.keras.layers.BatchNormalization()
        
       # self.conv4 = tf.keras.layers.Conv2D(32, strides=2)
       # self.bn5 = tf.keras.layers.BatchNormalization()
       # self.conv6 = tf.keras.layers.Conv2D(32, strides=2)
       # self.bn6 = tf.keras.layers.BatchNormalization()
        
       # self.conv7 = tf.keras.layers.Conv2D(64, strides=2)
        #self.bn7 = tf.keras.layers.BatchNormalization()
       # self.conv8 = tf.keras.layers.Conv2D(64, strides=2)
       # self.bn8 = tf.keras.layers.BatchNormalization()
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

        for m in self.get_config():
            if isinstance(m, tf.keras.layers.Conv2d):
                tf.keras.initializers.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (tf.keras.layers.BatchNormalization, tf.keras.layers.GroupNormalization)):
                tf.keras.initializers.Constant(m.weight, 1)
                tf.keras.initializers.Constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(planes, stride))
        for _ in range(1, blocks):
            layers.append(block(planes))

        return tf.keras.Sequential(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()
        x = self.fc(x)
     
        return x

class ResNet_SingleShared(tf.keras.layers.Layer):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=10):
        super(ResNet_SingleShared, self).__init__()
        self.in_planes = 16

        self.conv1 = tf.keras.layers.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.shared_basis_1 = tf.keras.layers.Conv2d(shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank*1, self.shared_basis_1, stride=1)
        
        self.shared_basis_2 = tf.keras.layers.Conv2d(shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
       
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, self.shared_basis_2, stride=2)
        
        self.shared_basis_3 = tf.keras.layers.Conv2d(shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
      
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, self.shared_basis_3, stride=2)
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D() 
        self.fc = tf.keras.layers.Dense(num_classes)
        
        for m in self.get_config():
            if isinstance(m, tf.keras.layers.Conv2d):
                tf.keras.initializers.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (tf.keras.layers.BatchNormalization, tf.keras.layers.GroupNormalization)):
                tf.keras.initializers.Constant(m.weight, 1)
                tf.keras.initializers.Constant(m.bias, 0)
                
        # Each share basis is orthogonal-initialized separately

        initializer=tf.orthogonal_initializer()
        
        initializer(self.shared_basis_1.weight)
        initializer(self.shared_basis_2.weight)
        initializer(self.shared_basis_3.weight)
        

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis, stride=1):
        layers = []
        
        layers.append(block_original(planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(planes, unique_rank, shared_basis))

        return tf.keras.Sequential(*layers)
    
    def _make_layer_original(self, block_original, planes, blocks, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_original(self.in_planes, planes, stride))

        return tf.keras.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()
        x = self.fc(x)
     
        return x

# Original ResNet
def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])

# A model with a single shared basis in each residual block group.
def ResNet32_SingleShared(shared_rank, unique_rank):
    return ResNet_SingleShared(BasicBlock_SingleShared, BasicBlock, [5, 5, 5], shared_rank, unique_rank)

