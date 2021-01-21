#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import Activation, InputLayer

__all__ = ['ConvolutionBlock', 'SeparableConvolutionBlock']

class ConvolutionBlock(Layer):
   """Constructs a generic convolution block for a convolutional neural network.

   This layer constructs a Convolution layer followed by a BatchNormalization layer. Activation either
   takes place after the Convolution layer or the BatchNormalization layer, by default after the convolution layer.

   Usage:

   The layer can be used in a Sequential model as would be any other layer.

   >>> model = Sequential()
   >>> model.add(ConvolutionBlock(32, kernel_size = (3, 3), activation = 'relu'))

   Or it can be used in a Functional model, as would be any other layer.

   >>> input = Input((256, 256, 3))
   >>> output = ConvolutionBlock(32, kernel_size = (3, 3), activation = 'softmax')(input)
   >>> model = Model(input, output)

   Parameters:
      - All of the parameters in the network are the same as those of the Conv2D & BatchNorm layers with a few left
        out because they are unnecessary in this circumstance.
      - Activation can either be directly following the convolution layer or after batch normalization,
        which can be set by use of the `activation_point` parameter: True means before, False means after.
   Returns:
      - A layer instance consisting of convolution, batch normalization, and activation.
   """
   def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'valid', activation = 'relu',
                activation_point = True, kernel_initializer = 'glorot_uniform', kernel_regularizer = None,
                kernel_constraint = None, use_bias = True, bias_initializer = 'zeros', bias_regularizer = None):
      super(ConvolutionBlock, self).__init__()

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
            self.batchnorm = BatchNormalization()
         else:
            # Build model with activation after.
            self.conv = Conv2D(
               filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm = BatchNormalization()
            self.activate = Activation(self.activation)
      except Exception as e:
         raise e

   def call(self, inputs, **kwargs):
      # When layer is called from model.
      x = self.conv(inputs)
      x = self.batchnorm(x)
      if not self.activation_point:
         x = self.activate(x)
      return x

class SeparableConvolutionBlock(Layer):
   """Constructs a depthwise separable convolution block for a convolutional neural network.

   This layer constructs a Depthwise Convolution layer, followed by BatchNormalization, then a
   Pointwise Convolution layer, that also followed by convolution. Activation either takes place 
   following each of the BatchNorms, or within the convolution layers, after by default.
   
   Usage:
   
   The layer can be used in a Sequential model as would be any other layer.

   >>> model = Sequential()
   >>> model.add(SeparableConvolutionBlock(32, kernel_size = (3, 3), activation = 'relu'))

   Or it can be used in a Functional model, as would be any other layer.

   >>> input = Input((256, 256, 3))
   >>> output = SeparableConvolutionBlock(32, kernel_size = (3, 3), activation = 'softmax')(input)
   >>> model = Model(input, output)
   
   Parameters:
      - The parameters are the same as the regular DepthwiseConv2D and Conv2D layers, only the filters 
        argument applies exclusively to the Conv2D layer and the kernel_size argument applies exclusively
        to the DepthwiseConv2D layer. All other parameters are shared between the layers.
      - Activation can be set to before BatchNorm layers, by setting `activation_point` to True. Otherwise
        it will take place after BatchNorm layers, as it will be set to False.
   Returns:
      - A layer instance consisting of a depthwise separable convolution.
   """
   def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'valid', activation = 'relu',
                activation_point = False, kernel_initializer = 'glorot_uniform', kernel_regularizer = None,
                kernel_constraint = None, use_bias = True, bias_initializer = 'zeros', bias_regularizer = None):
      super(SeparableConvolutionBlock, self).__init__()
      
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
            self.depthwise_conv = DepthwiseConv2D(
               kernel_size = kernel_size, strides = strides, padding = padding,
               activation = activation, kernel_initializer = kernel_initializer,
               kernel_regularizer = kernel_regularizer, kernel_constraint = kernel_constraint,
               use_bias = use_bias, bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm1 = BatchNormalization()
            self.pointwise_conv = Conv2D(
               filters = filters, kernel_size = (1, 1), strides = strides, padding = padding,
               activation = activation, kernel_initializer = kernel_initializer,
               kernel_regularizer = kernel_regularizer, kernel_constraint = kernel_constraint,
               use_bias = use_bias, bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm2 = BatchNormalization()
         else:
            self.depthwise_conv = DepthwiseConv2D(
               kernel_size = (1, 1), strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm1 = BatchNormalization()
            self.activation1 = Activation(self.activation)
            self.pointwise_conv = Conv2D(
               filters = filters, kernel_size = (1, 1), strides = strides, padding = padding,
               kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer,
               kernel_constraint = kernel_constraint, use_bias = use_bias,
               bias_initializer = bias_initializer, bias_regularizer = bias_regularizer
            )
            self.batchnorm2 = BatchNormalization()
            self.activation2 = Activation(self.activation)
      except Exception as e:
         raise e

   def call(self, inputs, **kwargs):
      # When layer is called from model.
      x = self.depthwise_conv(inputs)
      x = self.batchnorm1(x)
      if self.activation_point:
         x = self.activation1(x)
      x = self.pointwise_conv(x)
      x = self.batchnorm2(x)
      if self.activation_point:
         x = self.activation2(x)
      return x

