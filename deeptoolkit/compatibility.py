#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import re
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

from deeptoolkit.internal.conversion import convert_model_item
from deeptoolkit.internal.beautification import VALID_CONVERSION_LAYER_NAMES, LAYER_TORCH_CONVERSIONS
from deeptoolkit.internal.beautification import ACTIVATION_FUNCTION_NAMES

def _keras_to_torch_name(input_layer_name):
   """Internal method, helps to convert Keras layers to torch layers."""
   # Strip the ending numbers/underscores (we just need the layer name).
   name = re.sub("(_\\d+)", "", input_layer_name)

   # Reduce the name to lowercase.
   name = name.lower()

   # There are only a number of supported layers currently, so raise an error
   # if we encounter a layer that is currently not supported.
   if '1d' in name:
      raise ValueError(f"Received a one-dimensional layer {name}, currently "
                       f"1-D layers are not supported.")
   if '3d' in name:
      raise ValueError(f"Received a three-dimensional layer {name}, currently "
                       f"3-D layers are not supported.")
   elif name in ['add', 'concatenate']:
      raise ValueError("Multiple-branch models are not yet supported, so "
                       "add and concatenate layers will not work.")
   elif name not in VALID_CONVERSION_LAYER_NAMES:
      raise ValueError(f"Received an invalid layer name {name}, which is either "
                       f"incompatible with Keras/PyTorch or is invalid.")

   # Return the converted layer name.
   return LAYER_TORCH_CONVERSIONS[''.join(
      letter.upper() if indx == 0 else letter for indx, letter in enumerate(name))]

def _create_torch_layer(input_layer, input_layer_name = None,
                        _previous_layer_shape = None, _future_layer_shape = None):
   """Creates and instantiates a PyTorch layer."""
   # Get the layer object.

   # Get the dictionary containing the layer inputs.
   layer_params = input_layer.get_config()

   # Instantiate the layer for each different case.
   if 'conv' in input_layer_name.lower():
      # Create the Conv2d layer parameters.
      in_channels = _previous_layer_shape[1] if _previous_layer_shape[0] is None \
                                             else _previous_layer_shape[0]
      out_channels = layer_params['filters']
      kernel_size = layer_params['kernel_size']
      stride = layer_params['strides']
      if layer_params['padding'] == 'valid':
         padding = 0
      else:
         padding = 1

      # Create the Conv2d layer, keeping in mind the activation function.
      if layer_params['activation'] == 'linear':
         return nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                          kernel_size = kernel_size, stride = stride, padding = padding)
      else:
         return nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                          kernel_size = kernel_size, stride = stride, padding = padding), \
                _create_torch_layer(tf.keras.layers.Activation(layer_params['activation']),
                                    tf.keras.layers.Activation.__class__.__name__)
   if 'batchnorm' in input_layer_name.lower():
      # Create the BatchNormalization layer parameters.
      features = _previous_layer_shape[2] if _previous_layer_shape[0] is None \
                                          else _previous_layer_shape[1]
      return nn.BatchNorm2d(features, momentum = layer_params['momentum'])
   if 'maxpool' in input_layer_name.lower():
      # Create the MaxPooling layer parameters.
      if layer_params['padding'] == 'valid':
         padding = 0
      else:
         padding = 1
      return nn.MaxPool2d(kernel_size = layer_params['pool_size'],
                          stride = layer_params['strides'], padding = padding)
   if 'averagepool' in input_layer_name.lower():
      # Create the MaxPooling layer parameters.
      if layer_params['padding'] == 'valid':
         padding = 0
      else:
         padding = 1
      return nn.AvgPool2d(kernel_size = layer_params['pool_size'],
                          stride = layer_params['strides'], padding = padding)
   if 'activation' in input_layer_name.lower():
      # Create the Activation layer parameters.
      activation_function = layer_params['activation']
      activation_function = activation_function.replace('_', '').replace(' ', '').lower()
      return getattr(nn, ACTIVATION_FUNCTION_NAMES[activation_function])()
   if 'relu' in input_layer_name.lower():
      # If you are using a Keras ReLU Layer.
      return nn.ReLU()
   if 'softmax' in input_layer_name.lower():
      # If you are using a Keras Softmax Layer.
      return nn.Softmax()
   if 'dropout' in input_layer_name.lower():
      # Create the Dropout layer parameters.
      return nn.Dropout2d(p = layer_params['rate'])
   if 'linear' in input_layer_name.lower():
      # Create the Dense/Linear layer parameters.
      before_shape = _previous_layer_shape[1] if _previous_layer_shape[0] is None \
                                              else _previous_layer_shape[0]
      # Create the Dense/Linear layer, keeping in mind the activation function.
      if layer_params['activation'] == 'linear':
         return nn.Linear(before_shape, layer_params['units'])
      else:
         return nn.Linear(before_shape, layer_params['units']), \
                _create_torch_layer(tf.keras.layers.Activation(layer_params['activation']),
                                    tf.keras.layers.Activation.__name__)
   if 'flatten' in input_layer_name.lower():
      # Create the Flatten layer parameters.
      return nn.Flatten()

@convert_model_item
def keras_to_torch_architecture(model, **kwargs):
   """Converts a Keras model to a PyTorch model.

   Given a Keras model (or the path to a saved Keras model), this method
   converts it to a PyTorch nn.Sequential model, for usage in PyTorch functions.

   There are currently a number of restrictions on this method, however. It cannot
   accept branched models (e.g., it can only accept a sequential model), it cannot use
   1D/3D layers, only 2D layers, and it cannot take any recurrent layers. These will be
   implemented in the future, but should not be used with this method currently. If you
   want to build such a model, you will need to manually construct it.

   Most importantly, it currently cannot load model weights, so pretrained model
   conversions will not work, this is primarily for model architectures. This being said,
   it is still a useful method for converting various types of model architectures.

   Example:

   >>> model = tf.keras.models.load_model('my_model.hdf5')
   >>> torch_model = keras_to_torch_architecture(model)

   Arguments:
      - model: The actual model that you want to convert.
   """
   # Although the decorator should take care of this, this is more of a
   # simple helper validation for the upcoming steps in the method.
   if not isinstance(model, (tf.keras.models.Model, tf.keras.models.Sequential)):
      raise TypeError(f"Received invalid model type {type(model)}, expected a Keras model.")

   # Create a trackable list of the model layers.
   model_layers = []

   # Iterate over the Keras layers.
   for indx, layer in enumerate(model.layers):
      # We need to gather the input shape.
      if layer.__class__.__name__ in ['InputLayer']:
         # Get the input shape.
         continue

      # Get the proper layer name.
      _layer_name = _keras_to_torch_name(layer.__class__.__name__)

      # Get the previous layer (for conv2d compatibility)
      if indx > 0:
         _previous_layer_shape = model.layers[indx - 1].input_shape
      else:
         _previous_layer_shape = model.layers[0].input_shape[0]
      if isinstance(_previous_layer_shape, list):
         _previous_layer_shape = _previous_layer_shape[0]

      # Get the actual PyTorch layer.
      torch_layer = _create_torch_layer(layer, _layer_name, _previous_layer_shape = _previous_layer_shape)
      if isinstance(torch_layer, (list, tuple)):
         torch_layer = [item for item in torch_layer]
      else:
         torch_layer = [torch_layer]

      # Add it to the list of layers.
      model_layers.extend(torch_layer)

   # Create and return a sequential model from the layers.
   model = nn.Sequential(*model_layers)
   return model







