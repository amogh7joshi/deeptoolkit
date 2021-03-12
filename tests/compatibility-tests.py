#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import unittest

import torch.nn as nn
from tensorflow.keras.models import load_model

from deeptoolkit.compatibility import keras_to_torch_architecture

class CompatibilityModuleTest(unittest.TestCase):
   def test_keras_to_torch_architecture(self):
      """Ensure that the method converts the architectures correctly."""
      # Load the test Keras model.
      keras_model = load_model("files/keras_compatibility_test.h5")

      # Then, convert it to a torch architecture.
      torch_model = keras_to_torch_architecture(keras_model)

      # Check the model length and layers.
      self.assertEqual(len(torch_model), 21)
      self.assertTrue(isinstance(torch_model, nn.Sequential))
      for layer in torch_model:
         self.assertTrue(isinstance(layer, nn.Module))

      # Check a few layers.
      self.assertTrue(isinstance(torch_model[-1], nn.Softmax))
      self.assertTrue(isinstance(torch_model[0], nn.Conv2d))


