#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np
import tensorflow as tf

from deeptoolkit.internal.validation import validate_keras_model

@validate_keras_model
def unique_layers(model):
   """Return the unique layers from a Keras model.

   Given a model or the path to a model, this method will return a
   list containing the unique layer names from the model.

   Example:

   >>> model = tf.keras.models.Sequential([
   >>>   Dense(784, activation = 'relu'),
   >>>   Dropout(0.2),
   >>>   Dense(10, activation = 'softmax')
   >>>])
   >>> unique_layers(model)
   ... ['Dense', 'Dropout']

   Arguments:
      - model: A Keras model or the filepath to a weights file.
   """
   # Get the list of layer names and return the unique layers.
   return np.unique([layer.__class__.__name__ for layer in model.layers])

@validate_keras_model
def layer_counts(model):
   """Gets the counts of each unique layer from a Keras model.

   Given a model or the path to a model, this method will return a
   dictionary containing the unique layers in the model as well as
   the amount of each of these unique layers.

   Example:

   >>> model = tf.keras.models.Sequential([
   >>>   Dense(784, activation = 'relu'),
   >>>   Dropout(0.2),
   >>>   Dense(10, activation = 'softmax')
   >>>])
   >>> layer_counts(model)
   ... {'Dense': 2, 'Dropout': 1}

   Arguments:
      - model: A Keras model or a filepath to a weights file.
   """
   # Get the list of layer names.
   model_layers = [layer.__class__.__name__ for layer in model.layers]

   # Get the unique layer names.
   layer_names, counts = np.unique(model_layers, return_counts = True)

   # Return the dictionary containing the counts.
   return dict(zip(layer_names, counts))
