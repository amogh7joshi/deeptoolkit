#!/usr/bin/env python3
import unittest

import numpy as np
from deeptoolkit.data import train_val_test_split, shuffle_dataset

# Tests for the deeptoolkit.data module.

class DataModuleTest(unittest.TestCase):
   def test_data_split_lengths(self):
      """Ensure that train_val_test_split splits data into the correct amount of items."""
      X = np.random.random(100)
      y = np.random.random(100)
      X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, split = [0.6, 0.2, 0.2])
      self.assertEqual(len(X_train), 60)
      self.assertEqual(len(X_val), 20)
      self.assertEqual(len(X_test), 20)

   def test_shuffle_dataset(self):
      """Ensure that the dataset is able to shuffle properly without any errors, and
      contains the same number of items as before."""
      X = np.random.random(100)
      y = np.random.random(100)
      validation_X = []
      for item in X:
         validation_X.append(item)
      validation_X = np.array(validation_X)
      X, y = shuffle_dataset(X, y)
      self.assertEqual(len(np.union1d(X, validation_X)), 100)

if __name__ == '__main__':
   unittest.main()
