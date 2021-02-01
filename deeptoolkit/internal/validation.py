#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import functools

import numpy as np
import pandas as pd

# Decorator Methods for Internal Use.

def validate_data_shapes(func):
   """Decorator to validate data shapes.

   Used on functions in the `deeptoolkit.data` module, to ensure that
   data formats are properly inputted, both to make sure that the function doesn't break
   with invalid inputs, but also to warn the user in case of invalid data.
   """
   @functools.wraps(func)
   def inner_decorator(*args, **kwargs):
      """Validate data shapes."""
      data_shape = np.array(args[0]).shape
      converted_args = []
      for item in args:
         if isinstance(item, (list, tuple)):
            item = np.array(item)
         if func.__name__ not in ['plot_class_distribution']:
            if not item.shape == data_shape:
               raise ValueError(f"All dataset items provided to {func.__name__} must have the same length, "
                                f"got items with length {data_shape} and {item.shape}.")
         converted_args.append(item)
      return func(*converted_args, **kwargs)
   return inner_decorator

def convert_log_item(func):
   """Decorator to validate data log item.

   Used on functions in the `deeptoolkit.evaluation` module, to ensure that either data paths
   are provided directly, or the data format is provided correctly.
   """
   @functools.wraps(func)
   def inner_decorator(*args, **kwargs):
      """Validate log item."""
      if len(args) < 1:
         raise ValueError(f"Got no arguments for function {func.__name__}, expected either a single or "
                          f"multiple arguments,  each being either the training path or history item.")
      if func.__name__ == 'plot_training_curves' and len(args) != 1:
         raise ValueError(f"Got multiple arguments for function {func.__name__}, expected only "
                          f"a single argument, either the training path or history item.")

      # Create list of converted arguments.
      converted_args = []

      # Convert each individual argument.
      for argument in args:
         if isinstance(argument, pd.DataFrame):
            try:
               # First try to iterate over the passed argument.
               iter(argument)
            except TypeError:
               try:
                  # Determine if object has attribute 'history', as is the regular Keras model.history
                  # object. If so, then try to iterate over it, if it can't be, then there's an error.
                  iter(argument.history)
               except Exception as e:
                  raise e
               else:
                  history = argument.history
            except Exception as e:
               raise e
            else:
               history = argument
         elif os.path.exists(argument):
            try:
               # Try to read the training log if a file was passed.
               history = pd.read_csv(argument)
            except Exception as e:
               print(f"Got path argument for {func.__name__}, but encountered error while trying to read it. "
                     f"Check your path and ensure that it is a csv file containing training logs.")
               raise e
         else:
            raise TypeError(f"Invalid argument provided, got {type(argument)} while expecting either "
                            f"a training log csv path or a DataFrame object. ")

         converted_args.append(history)

      # If function is plot_training_curves, only return the single argument, otherwise return all.
      if func.__name__ == 'plot_training_curves':
         return func(converted_args[0], **kwargs)
      else:
         return func(*converted_args, **kwargs)
   return inner_decorator




