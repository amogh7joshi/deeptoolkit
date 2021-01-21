#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import functools

import numpy as np
import pandas as pd

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
      if len(args) != 1:
         raise ValueError(f"Got multiple arguments for function {func.__name__}, expected only "
                          f"a single argument, the training path or history item.")
      argument = args[0]
      if os.path.exists(argument):
         try:
            history = pd.read_csv(argument)
         except Exception as e:
            print(f"Got path argument for {func.__name__}, but encountered error while trying to read it. "
                  f"Check your path and ensure that it is a csv file containing training logs.")
            raise e
      else:
         try:
            for _ in argument:
               pass
         except TypeError:
            try:
               for _ in argument.history:
                  pass
            except Exception as e:
               raise e
            else:
               history = argument.history
         else:
            history = argument

      # Return function with evaluated history argument.
      return func(history, **kwargs)
   return inner_decorator



