#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from deeptoolkit.internal.validation import validate_data_shapes


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
   plt.figure()

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








