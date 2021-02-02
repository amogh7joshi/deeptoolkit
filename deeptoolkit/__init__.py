#!/usr/bin/env python3
# -*- coding = utf-8 -*-
"""
DeepToolKit

This package provides implementations of popular machine learning algorithms, extensions to existing
deep learning pipelines using TensorFlow and Keras, and convenience utilities to speed up the process
of implementing, training, and testing deep learning models.
"""
from __future__ import absolute_import

import os
import sys
import logging
import warnings
import setuptools

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Import DeepToolKit functions.
from deeptoolkit.internal.acquisition import load_dnn_files

# Set up system validation (with warnings and logging).
warnings.filterwarnings('default')
logging.basicConfig(format = '%(levelname)s - %(name)s: %(message)s')

if not K.backend() == 'tensorflow':
   logging.warning(f"Your backend is {K.backend()}, which is not fully compatible with DeepToolKit. This package is built "
                   f"on a TensorFlow backend, so please switch your Keras backend to TensorFlow for optimal performance.")

if tf.__version__ < '2.0.0':
   logging.warning(f"You are using TensorFlow with version < 2.x.x, which may cause issues in the DeepToolKit library. "
                   f"Please consider upgrading to TensorFlow 2.x.x for optimal DeepToolKit performance.")

def validate_image_format():
   if K.image_data_format() == 'channels_last':
      # Using the correct backend.
      pass
   elif K.image_data_format() == 'channels_first':
      # Using a non-TensorFlow backend, which may cause the package to break.
      logging.warning("Your image_data_format is set to channels_first, implying you are using a Theano or other "
                      "backend for Keras. This package is specifically built for a TensorFlow backend to Keras, "
                      "and many utilities may not function properly with a non-TensorFlow backend. Please switch your "
                      "backend to TensorFlow for optimal DeepToolKit usage.")
   else:
      # Using an unknown image_data_format.
      raise ValueError(f"Your image_data_format is {K.image_data_format()}, which is an unrecognized image_data_format. "
                       f"Valid image_data_format(s) are channels_first, and channels_last. Check your backend and your "
                       f"TensorFlow/Keras configuration, so that the package does not break.")

# Validate image data format.
validate_image_format()

# The following methods construct lists of valid module objects.
# From the individual modules, such as deeptoolkit.blocks or deeptoolkit.losses,
# the __all__ object is accessed and displayed here when called.

def list_valid_blocks() -> list:
   """Construct and display list of valid layer blocks in DeepToolKit."""
   import deeptoolkit.blocks as _blocks
   displayable_list = ', '.join(item for item in _blocks.__all__)
   del _blocks # To prevent runaway imports.
   print("Valid layer blocks in deeptoolkit.blocks: " + displayable_list)
   return displayable_list.split(', ')

def list_valid_losses() -> list:
   """Construct and display list of valid layer blocks in DeepToolKit."""
   import deeptoolkit.losses as _losses
   displayable_list = ', '.join(item for item in _losses.__all__)
   del _losses # To prevent runaway imports.
   print("Valid loss functions in deeptoolkit.losses: " + displayable_list)
   return displayable_list.split(', ')

# Create the __all__ attribute for the top-level deeptoolkit module.
__all__ = []

# Add list of primary modules.
modules = list(setuptools.find_packages())
__all__.extend(modules)

# Add list of top-level methods.
top_level_methods = ['list_valid_blocks', 'list_valid_losses', 'load_dnn_files']
top_level_methods = [f'deeptoolkit.{method}' for method in top_level_methods]
__all__.extend(top_level_methods)




