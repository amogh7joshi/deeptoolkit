#!/usr/bin/env python3
# -*- coding = utf-8 -*-
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
      """Apply tensor conversion"""
      converted_args = []
      for item in args:
         if isinstance(item, (tf.keras.losses.Loss, tf.keras.metrics.Metric)):
            converted_args.append(item)
            continue
         if not isinstance(item, (list, tuple, np.ndarray)):
            raise TypeError(f"Items provided to a loss function should be either a list, tuple, or numpy array, "
                            f"got item of type {type(item)}.")
         else:
            item = tf.convert_to_tensor(item, tf.float32)
         converted_args.append(item)
      return func(*converted_args, **kwargs)
   return inner_decorator
