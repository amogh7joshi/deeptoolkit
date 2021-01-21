#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from deeptoolkit.internal.validation import convert_log_item
from deeptoolkit.internal.beautification import LOG_DICT_NAMES

@convert_log_item
def plot_training_curves(history, metrics = None, save = False):
   """Plot training curves from model training history.

   This method plots the training metric curves over time from a model training session,
   provided either the log object or the path to a csv file containing the data. The metrics
   to plot should be provided in the `metrics` argument.

   Parameters:
      - history: A training history object, or the path to a csv containing training history.
      - metrics: The metrics you want to plot.
      - save: Whether to save the image to an image file.
   """
   # Validate metrics.
   valid_metrics = []
   for item in history:
      if item != 'Unnamed: 0':
         valid_metrics.append(item)
   if not metrics:
      raise ValueError("You need to provide some metrics to plot, otherwise there is nothing to plot.")
   for metric in metrics:
      if metric not in history:
         raise ValueError(f"Got invalid metric {metric}. Valid metrics for the training session provided are {valid_metrics}.")

   # Plot metrics.
   plt.xlabel('epoch')
   for item in metrics:
      plt.plot(history[item], label = LOG_DICT_NAMES[item])
   plt.legend(loc = 'upper left')
   savefig = plt.gcf()
   plt.show()

   # Save image if it states to save.
   if save:
      try:
         savefig.savefig(save)
      except Exception as e:
         if not isinstance(save, str):
            raise ValueError("If you want to save the image, you need to provide a save path for the `save` argument.")
         else:
            raise e




