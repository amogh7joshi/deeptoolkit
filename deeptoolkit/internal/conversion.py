#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import re
import functools

import numpy as np
import tensorflow as tf

def apply_tensor_conversion(func):
   """Decorator to convert arguments to tensors.

   Used primarily on loss functions in the `deeptoolkit.losses` module, to convert
   arguments to tf.Tensors for compatibility in the actual internal loss function.
   """
   @functools.wraps(func)
   def inner_decorator(*args, **kwargs):
      """Apply tensor conversion."""
      converted_args = []
      for item in args:
         if isinstance(item, (tf.keras.losses.Loss, tf.keras.metrics.Metric)):
            converted_args.append(item)
            continue
         if not isinstance(item, (list, tuple, np.ndarray)):
            raise TypeError(f"Items provided to a loss function should be "
                            f"either a list, tuple, or numpy array, "
                            f"got item of type {type(item)}.")
         else:
            item = tf.convert_to_tensor(item, tf.float32)
         converted_args.append(item)
      return func(*converted_args, **kwargs)
   return inner_decorator

def convert_model_item(func):
   """Decorator to convert arguments to Keras/TensorFlow models.

   Used primarily on conversion methods in the `deeptoolkit.compatibility` module, to
   convert inputted saved model files or paths to models.
   """
   @functools.wraps(func)
   def inner_decorator(*args, **kwargs):
      """Convert a saved model item."""
      model_item = args[0]
      converted_args = list(args[1:])
      if isinstance(model_item, (tf.keras.models.Model, tf.keras.models.Sequential)):
         # Provided model item is already a Keras model.
         converted_args.insert(0, model_item)
      elif isinstance(model_item, str):
         # Provided model item is a string.
         if os.path.exists(model_item):
            # Provided model item is a valid path.
            try:
               # The model might need custom objects.
               model = tf.keras.models.load_model(model_item)
               converted_args.insert(0, model)
            except ValueError as ve:
               # Determine the custom objects.
               if re.search('unknown', ve.__str__().lower()):
                  # Check if custom objects have been provided.
                  if 'custom_objects' in kwargs:
                     try:
                        # Load the model with custom objects.
                        model = tf.keras.models.load_model(
                           model_item, custom_objects = kwargs['custom_objects'])
                        converted_args.insert(0, model)
                     except Exception as e:
                        # Unknown exception has occurred.
                        raise e
                  else:
                     # Custom objects must have been provided otherwise this won't work.
                     # Get the list of custom objects.
                     raise ValueError("The provided model expects custom objects: {}".format(
                        [item.strip() for item in re.findall(':\\s(.*?)$', ve.__str__())]))
            except Exception as e:
               # Unknown exception has occurred.
               raise e
         else:
            # Did not receive a path.
            raise FileNotFoundError(f"Received a path {model_item}, but the path does not exist.")
      else:
         # Received an invalid type.
         raise TypeError(f"Expected either a path to a model or an actual Model, got {type(model_item)}.")

      # Return the function.
      return func(*tuple(converted_args), **kwargs)
   return inner_decorator




