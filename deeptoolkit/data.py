#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import

import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as _train_test_split

from deeptoolkit.internal.validation import validate_data_shapes

__all__ = ['train_val_test_split', 'shuffle_dataset', 'reduce_dataset', 'plot_data_cluster', 'plot_class_distribution']

@validate_data_shapes
def train_val_test_split(X, y, *, split: Any = 0.7):
   """Split datasets into train, validation, and testing sets.

   Given training data X and training labels y, the method will split the data
   into relevant training, validation, and testing sets based on the `split` parameter.

   Usage:

   The method can be called directly with training data.

   >>> X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
   >>> y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   >>> X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, split = [0.6, 0.2, 0.2])

   Parameters:
      - X: The training data, should be lists or arrays.
      - y: The training labels, should be lists or arrays.
      - split: How you want to split the data. Either a float m, which will represent the percentage of
               training data, and the val/test data will have a percentage (1 - m)/2, or a list of three numbers,
               containing the exact float percentages for train/val/test data. Defaults to 70/15/15 split.
   Returns:
      - Six arrays: training data, validation data, test data, train labels, validation labels, test labels.
   """
   # Verify and segment provided split.
   if isinstance(split, float):
      train_split = split
      val_split = test_split = (1 - split) / 2
   elif isinstance(split, (list, tuple)):
      if not len(split) == 3:
         raise ValueError("If you are providing a list of percentages for the train/val/test split, it "
                          f"must contain three numbers, got {len(split)}.")
      train_split = split[0]; val_split = split[1]; test_split = split[2]
      if not train_split + val_split + test_split == 1:
         raise ValueError("If you are providing a list of percentages for the train/va/test split, it "
                          f"should add up to 1, got {train_split + val_split + test_split}")
   else:
      raise TypeError("Split argument should either be a float representing the training percentage, "
                      f"or a list containing the train/val/test percentages, got {type(split)}.")

   # Convert validation/test split to relative numbers.
   total_test_val_split = val_split + test_split
   val_split = val_split / total_test_val_split

   # First, convert data to train/overflow.
   X_train, X_overflow, y_train, y_overflow = _train_test_split(X, y, train_size = train_split)

   # Then, convert overflow to val/test.
   X_val, X_test, y_val, y_test = _train_test_split(X_overflow, y_overflow, train_size = val_split)

   # Finally, return split training, validation, and test data.
   return X_train, X_val, X_test, y_train, y_val, y_test

@validate_data_shapes
def shuffle_dataset(*args):
   """Shuffle inputted datasets randomly.

   Given a number of pieces of training data/training labels, this method will
   shuffle the data. The mapping between pieces of training data and training labels
   will be retained, as a single randomization will be used for every piece of data.

   Usage:

   The method can be called directly with training data.
   >>> X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
   >>> y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   >>> X, y = shuffle_dataset(X, y)

   Parameters:
      - args: The arrays that you want to shuffle.
   Returns:
      - The initial arguments with shuffled orders, converted to numpy arrays.
   """
   # Create random number list.
   data_shape = args[0].shape[0]
   random_list = np.random.choice(data_shape, data_shape, replace = False)

   # Apply to inputted arrays.
   shuffled_items = []
   for item in args:
      current_list = []
      for value in random_list:
         current_list.append(item[value])
      current_list = np.array(current_list)
      shuffled_items.append(current_list)

   # Return individual item if only one is provided, else return list of all items.
   if len(shuffled_items) == 1:
      return shuffled_items[0]
   return shuffled_items

@validate_data_shapes
def reduce_dataset(*args, reduction = None, shuffle = True):
   """Reduce a dataset to a certain number of items.

   Given a number of pieces of training data/training labels, this method will return a
   reduced portion of the data, and optionally shuffle the data before it reduces it.
   The mapping between pieces of training data and training labels will be retained, as a
   single randomization will be used for every piece of data.

   Usage:

   The method can be called directly with training data.
   >>> X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
   >>> y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   >>> X_reduced, y_reduced = reduce_dataset(X, y)

   Parameters:
      - args: The arrays that you want to reduce.
      - reduction: The number of items or the percentage of items you want to get back, defaults to 10%.
      - shuffle: Whether to return a random portion of the dataset or the first m items, default True.
   Returns:
      - Reduced, smaller, and potentially shuffled versions of the inputted arrays.
   """
   # Determine data shape.
   data_shape = args[0].shape[0]

   # Validate reduction argument.
   if reduction is None:
      reduction = data_shape // 10
   if isinstance(reduction, float):
      reduction = int(reduction * data_shape)
   if reduction > data_shape:
      raise ValueError(f"You have provided a length longer than that of the dataset: {reduction} > {data_shape}.")

   # If requested to, shuffle items first.
   initial_items = args
   if shuffle:
      initial_items = shuffle_dataset(initial_items)

   # Reduce dataset.
   reduced_items = []
   for item in initial_items:
      reduced_items.append(item[:reduction])

   # Return individual item if only one is provided, else return list of all items.
   if len(reduced_items) == 1:
      return reduced_items[0]
   return reduced_items


@validate_data_shapes
def plot_data_cluster(X, y, classes, *, save = False):
   """Create a scatter plot of a data cluster.

   Given training data X and training labels y, where `classes` corresponds to
   the class label of y, this method creates a scatter plot visualization, and
   saves it to an image file if requested to.

   Usage:

   This method can be called directly with training data.

   >>> X = np.random.random(100)
   >>> y = np.random.random(100)
   >>> labels = np.random.choice(range(3), 100)
   >>> plot_data_cluster(X, y, labels)

   Arguments:
      - X: Training data
      - y: Training labels
      - classes: The class labels for each class, for a maximum of 10 different class labels.
                 The method then makes a color map out of these labels.
      - save: Whether you want to save the figure to an image file. If you do, then input the
              image path as the value for this argument.
   """
   # Create figure.
   fig = plt.figure()
   ax = fig.add_subplot(111)

   # Scatter data onto figure.
   ax.scatter(X, y, c = classes, lw = 0)

   # Display figure.
   savefig = plt.gcf()
   plt.show()

   # If requested to, save image.
   if save:
      try:
         savefig.savefig(save)
      except Exception as e:
         if not isinstance(save, str):
            raise ValueError("If you want to save the image, you need to provide a save path for the `save` argument.")
         else:
            raise e

@validate_data_shapes
def plot_class_distribution(*args, save = False):
   """Create a plot visualizing the class distribution of provided arrays.

   Given an arbitrary number of arrays, this method will construct and display a bar plot
   showing the frequency of each class item.

   Usage:

   >>> y = [np.random.randint(0, 50) for _ in range(101)]
   >>> y_train = y[:60]
   >>> y_val = y[61:80]
   >>> y_test = y[81:]
   >>> plot_class_distribution(y_train, y_val, y_test)

   Arguments:
      - args: The arrays from which the class distribution will be constructed.
      - save: Whether you want to save the figure to an image file. If you do, then input the
              image path as the value for this argument.
   """
   # Create figure.
   fig = plt.figure()

   for item in args:
      try:
         # Plot a bar plot of each of the individual arrays provided.
         # If there was only supposed to be one array, this will still only plot one.
         unique, counts = np.unique(item, return_counts = True)
         plt.bar(unique, counts)
      except Exception as e:
         raise e
      finally:
         # Delete the items regardless to clear up the variables for the next iteration.
         del unique, counts

   # Set up the rest of the figure.
   plt.title('Class Frequency')
   plt.xlabel('Class')
   plt.ylabel('Frequency')

   # Display figure.
   savefig = plt.gcf()
   plt.show()

   # If requested to, save image.
   if save:
      try:
         savefig.savefig(save)
      except Exception as e:
         if not isinstance(save, str):
            raise ValueError("If you want to save the image, you need to provide a save path for the `save` argument.")
         else:
            raise e







