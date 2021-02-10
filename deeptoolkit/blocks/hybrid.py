#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation, InputLayer, Concatenate

__all__ = ['HybridConvolution']

class HybridConvolution(Layer):
   """Constructs a hybrid convolution block for use in a convolutional neural network.

   This layer constructs two individual Convolution and Separable Convolution layers, and then concatenates
   them together, forming a hybrid convolution layer, for use in deeper model architectures.

   Usage:

   The layer can be used in a Sequential model as would be any other layer.

   >>> model = Sequential()
   >>> model.add(HybridConvolution(32, kernel_size = (3, 3), activation = 'relu'))

   Or it can be used in a Functional model, as would be any other layer.

   >>> input = Input((256, 256, 3))
   >>> output = HybridConvolution(32, kernel_size = (3, 3), activation = 'softmax')(input)
   >>> model = Model(input, output)

   Parameters:
      - All of the parameters in the network are the same as those of the Conv2D/SeparableConv2D & BatchNorm
        layers with a few left out because they are unnecessary in this circumstance.
      - Activation can either be directly following the convolution layer or after batch normalization,
        which can be set by use of the `activation_point` parameter: True means before, False means after.
   Returns:
      - A layer instance consisting of convolution, batch normalization, and activation.
   """
   def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'valid', activation = 'relu',
                activation_point = True, kernel_initializer = 'glorot_uniform', kernel_regularizer = None,
                kernel_constraint = None, use_bias = True, bias_initializer = 'zeros', bias_regularizer = None):
      super(HybridConvolution, self).__init__()

      # Set class parameters.
      self.filters = filters
      if isinstance(kernel_size, int):
         self.kernel_size = (kernel_size, kernel_size)
      else:
         self.kernel_size = kernel_size
      if isinstance(strides, int):
         self.strides = (strides, strides)
      else:
         self.strides = strides
      self.padding = padding
      self.activation = activation
      self.activation_point = activation_point
      self.kernel_initializer = kernel_initializer
      self.kernel_regularizer = kernel_regularizer,
      self.kernel_constraint = kernel_constraint,
      self.use_bias = use_bias
      self.bias_initializer = bias_initializer
      self.bias_regularizer = bias_regularizer

      # Set class layers.
      try:
         if self.activation_point:
            # Build model with activation first.
            self.conv = Conv2D(
               filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
               activation = activation, kernel_initializer = kernel_initializer,
               kernel_regularizer = kernel_regularizer, kernel_constraint = kernel_constraint,
               use_bias = use_bias, bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm1 = BatchNormalization()
            self.separable_conv = SeparableConv2D(
               filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
               activation = activation, kernel_initializer = kernel_initializer,
               kernel_regularizer = kernel_regularizer, kernel_constraint = kernel_constraint,
               use_bias = use_bias, bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
         else:
            # Build model with activation after.
            self.conv = Conv2D(
               filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm1 = BatchNormalization()
            self.activate1 = Activation(self.activation)
            self.separable_conv = SeparableConv2D(
               filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm2 = BatchNormalization()
            self.activate2 = Activation(self.activation)
      except Exception as e:
         raise e

   def call(self, inputs, **kwargs):
      # When layer is called from model.
      # Vanilla convolution branch.
      conv_x = self.conv(inputs)
      conv_x = self.batchnorm1(conv_x)
      if not self.activation_point:
         conv_x = self.activate1(conv_x)

      # Depthwise separable convolution branch.
      sep_conv_x = self.separable_conv(inputs)
      sep_conv_x = self.batchnorm2(sep_conv_x)
      if not self.activation_point:
         sep_conv_x = self.activate2(sep_conv_x)

      # Merge branches and return.
      x = Concatenate()([conv_x, sep_conv_x])
      return x

class HybridPooling(Layer):
   """Constructs a hybrid convolution block for use in a convolutional neural network.

   This layer constructs two individual Convolution and Separable Convolution layers, and then concatenates
   them together, forming a hybrid convolution layer, for use in deeper model architectures.

   Usage:

   The layer can be used in a Sequential model as would be any other layer.

   >>> model = Sequential()
   >>> model.add(HybridConvolution(32, kernel_size = (3, 3), activation = 'relu'))

   Or it can be used in a Functional model, as would be any other layer.

   >>> input = Input((256, 256, 3))
   >>> output = HybridPooling(pool_size = (2, 2))(input)
   >>> model = Model(input, output)

   Parameters:
      - All of the parameters in the network are the same as those of the Conv2D/SeparableConv2D & BatchNorm
        layers with a few left out because they are unnecessary in this circumstance.
      - Activation can either be directly following the convolution layer or after batch normalization,
        which can be set by use of the `activation_point` parameter: True means before, False means after.
   Returns:
      - A layer instance consisting of convolution, batch normalization, and activation.
   """
   def __init__(self, pool_size = (2, 2), strides = (1, 1), padding = 'valid'):
      super(HybridPooling, self).__init__()

      # Set class parameters.
      if isinstance(pool_size, int):
         self.pool_size = (pool_size, pool_size)
      else:
         self.pool_size = pool_size
      if isinstance(strides, int):
         self.strides = (strides, strides)
      else:
         self.strides = strides
      self.padding = padding

      # Set class layers.
      try:
         self.max_pool = MaxPooling2D(pool_size = pool_size, strides = strides, padding = padding)
         self.avg_pool = AveragePooling2D(pool_size = pool_size, strides = strides, padding = padding)
      except Exception as e:
         raise e

   def call(self, inputs, **kwargs):
      # When layer is called from model.
      # Max pooling branch.
      max_pool_x = self.max_pool(inputs)

      # Average pooling branch.
      avg_pool_x = self.avg_pool(inputs)

      # Merge branches and return.
      x = Concatenate()([max_pool_x, avg_pool_x])
      return x
