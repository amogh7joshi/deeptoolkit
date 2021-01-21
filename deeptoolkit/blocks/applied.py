#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Reshape, Dense, Multiply, Add, Activation

from deeptoolkit.blocks.generic import ConvolutionBlock

__all__ = ['SqueezeExcitationBlock', 'ResNetIdentityBlock']

class SqueezeExcitationBlock(Layer):
   """Constructs a generic squeeze-excitation block.

   This layer constructs a squeeze-excitation block as defined in the paper
   `Squeeze-And-Excitation Networks`: https://arxiv.org/abs/1709.01507. It consists of
   a GlobalAveragePooling layer, followed by two fully-connected layers. Generally, it
   should be implemented in an SENet architecture as a residual connection.

   Usage:

   The layer can be used in a Sequential model as would be any other layer.

   >>> model = Sequential()
   >>> model.add(SqueezeExcitationBlock(ratio = 0.5))

   Or it can be used in a Functional model, as would be any other layer.

   >>> input = Input((256, 256, 3))
   >>> output = SqueezeExcitationBlock(ratio = 0.5)(input)
   >>> model = Model(input, output)

   Parameters:
      - ratio: The number of output channels in the final image.
   Returns:
      - A squeeze-and-excitation layer.
   """
   def __init__(self, ratio):
      super(SqueezeExcitationBlock, self).__init__()

      # Set up class variables.
      self.ratio = ratio

      # No class layers set except for GlobalAveragePooling2D because they are all
      # reliant on input channels, which are only provided when layer is called.
      self.pooling = GlobalAveragePooling2D()

   def call(self, inputs, **kwargs):
      # Determine input channels.
      input_channels = int(inputs.shape[-1])

      # When layer is called by model.
      x = self.pooling(inputs)
      x = Reshape((1, 1, input_channels))(x)
      x = Dense(input_channels // self.ratio, activation = 'relu',
                kernel_initializer = 'he_normal', use_bias = False)(x)
      x = Dense(input_channels, activation = 'sigmoid',
                kernel_initializer = 'he_normal', use_bias = False)(x)
      x = Multiply()([inputs, x])
      return x

class ResNetIdentityBlock(Layer):
   """Constructs a generic ResNet Identity Block layer.

   This layer constructs the generic ResNet identity block as defined in the paper
   `Deep Residual Learning for Image Recognition`: https://arxiv.org/abs/1512.03385.
   It consists of two Convolution Blocks (see deeptoolkit.blocks.generic.ConvolutionBlock),
   followed by convolution, BatchNormalization, and addition with a shortcut layer representing
   a residual connection, following by final activation.

   Parameters:
      - A majority of the parameters are the same as in Conv2D, but the different ones are listed below.
      - filters: Either an integer m, at which point the first & second convolution will have filters m,
                 and the third convolution will have filters m * 4, or a list/tuple of length 3, where each
                 argument is the filters for a convolution block.
      - kernel_size: Serves the same purpose as in Conv2D, but only applies to the second convolution.
      - conv_shortcut: Used when tensors may have different shape. If set to True, the shortcut tensor will
                       go through a convolution shortcut, otherwise through the simple identity shortcut.
   Returns:
      - A layer instance consisting of a ResNet identity block.
   """
   def __init__(self, filters, kernel_size, conv_shortcut = True, strides = (1, 1), padding = 'valid',
                activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = None,
                kernel_constraint = None, use_bias = True, bias_initializer = 'zeros', bias_regularizer = None):
      super(ResNetIdentityBlock, self).__init__()

      # Set class parameters.
      if isinstance(filters, int):
         filters = (filters, filters, 4 * filters)
      elif isinstance(filters, (list, tuple)):
         if not len(filters) == 3:
            raise ValueError(f"If you are providing a list of filters, it should have "
                             f"length 3, got length {len(filters)}.")
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
      self.kernel_initializer = kernel_initializer
      self.kernel_regularizer = kernel_regularizer,
      self.kernel_constraint = kernel_constraint,
      self.use_bias = use_bias
      self.bias_initializer = bias_initializer
      self.bias_regularizer = bias_regularizer
      self.conv_shortcut = conv_shortcut

      # Set class layers.
      try:
         if self.conv_shortcut:
            self.shortcut_conv = Conv2D(
               filters = filters[2], kernel_size = (1, 1), strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.shortcut_batchnorm = BatchNormalization()

         # Set remaining layers.
         self.conv_block_1 = ConvolutionBlock(
            filters = filters[0], kernel_size = (1, 1), strides = strides, padding = padding,
            kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
            kernel_constraint = kernel_constraint, use_bias = use_bias, activation_point = False,
            bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
         )
         self.conv_block_2 = Conv2D(
            filters = filters[1], kernel_size = kernel_size, strides = strides, padding = padding,
            kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
            kernel_constraint = kernel_constraint, use_bias = use_bias, activation_point = False,
            bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
         )
         self.conv3 = Conv2D(
            filters = filters[2], kernel_size = (1, 1), strides = strides, padding = padding,
            kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
            kernel_constraint = kernel_constraint, use_bias = use_bias,
            bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
         )
         self.batchnorm3 = BatchNormalization()
      except Exception as e:
         raise e

   def call(self, inputs, **kwargs):
      # When layer is called by model.
      x_shortcut = inputs
      if self.conv_shortcut: # Shortcut convolution identity block.
         x_shortcut = self.shortcut_conv(x_shortcut)
         x_shortcut = self.shortcut_batchnorm(x_shortcut)

      # Primary system.
      x = self.conv_block_1(inputs)
      x = self.conv_block_2(x)
      x = self.conv3(x)
      x = self.batchnorm3(x)

      # Bring inputs together and return final layer.
      x = Add()([x_shortcut, x])
      x = Activation(self.activation)(x)
      return x

