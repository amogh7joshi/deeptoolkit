#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

class DefaultDictionary(dict):
   """A special dictionary to return the default key if it is missing."""
   def __missing__(self, key):
      return key

def select_threshold(tensor, thresh = None):
   """Returns a tensor with items greater than the provided threshold set to 1, else 0."""
   if thresh is None: # If threshold is not provided.
      return tensor
   tensor = K.greater(tensor, thresh)
   tensor = tf.cast(tensor, tf.float32)
   return tensor


