#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from deeptoolkit.internal.validation import convert_log_item
from deeptoolkit.internal.beautification import LOG_DICT_NAMES

__all__ = ['plot_training_curves', 'plot_multiple_training_curves', 'concat_training_logs']

@convert_log_item
def plot_training_curves(history, metrics = 'default', save = False):
   """Plot training curves from model training history.

   This method plots the training metric curves over time from a model training session,
   provided either the log object or the path to a csv file containing the data. The metrics
   to plot should be provided in the `metrics` argument.

   Parameters:
      - history: A training history object, or the path to a csv containing training history.
      - metrics: The metrics you want to plot.
      - save: Whether you want to save the figure to an image file. If you do, then input the
              image path as the value for this argument.
   """
   # Validate metrics.
   valid_metrics = []
   for item in history:
      if 'Unnamed' not in item:
         valid_metrics.append(item)
   if not metrics:
      raise ValueError("You need to provide some metrics to plot, otherwise there is nothing to plot.")
   if metrics == 'default':
      if 'acc' in history:
         metrics = ['acc', 'val_acc', 'loss', 'val_loss']
      elif 'accuracy' in history:
         metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
      else:
         raise ValueError("Can't use default metrics when acc/accuracy is not in training session metrics. "
                          f"Valid metrics for the training session provided are {valid_metrics}. ")
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

@convert_log_item
def plot_multiple_training_curves(*objects, metrics = 'default', save = False):
   """Plot multiple training curves from model training history.

   This method plots multiple subplots showing the training metric curves over time from each
   provided model training session log, either the log object or the path to a csv file containing the data.
   The metrics to plot should be provided in the `metrics` argument.

   Parameters:
      - objects: Training history objects or csv log files.
      - metrics: The metrics you want to plot.
      - save: Whether you want to save the figure to an image file. If you do, then input the
              image path as the value for this argument.
   """
   # Validate metrics.
   for history in objects:
      # Track the valid metrics.
      valid_metrics = []
      for item in history:
         if 'Unnamed' not in item:
            valid_metrics.append(item)
      if not metrics:
         raise ValueError("You need to provide some metrics to plot, otherwise there is nothing to plot.")
      if metrics == 'default':
         if 'acc' in history:
            metrics = ['acc', 'val_acc', 'loss', 'val_loss']
         elif 'accuracy' in history:
            metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
         else:
            raise ValueError("Can't use default metrics when acc/accuracy is not in training session metrics. "
                             f"Valid metrics for the training session provided are {valid_metrics}. ")

         for metric in metrics:
            if metric not in history:
               raise ValueError(f"Got invalid metric {metric}. Valid metrics for the training "
                                f"session provided are {valid_metrics}.")

   # Plot metrics.
   fig, axes = plt.subplots(1, len(objects), figsize = (6 * len(objects), 6))
   for indx, (history, ax) in enumerate(zip(objects, axes)):
      ax.set_xlabel('epoch')
      ax.set_title(f'Plot {indx + 1}')
      for item in metrics:
         ax.plot(history[item], label = LOG_DICT_NAMES[item])
      ax.legend(loc = 'upper left')

   # Display the figure.
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

@convert_log_item
def concat_training_logs(*logs, save = False):
   """Concatenates a set of provided training logs into a single complete training log.

   Provided a number of individual training logs in order for the same model, for example if
   training has been stopped and restarted, this method will concatenate all of the logs
   into a single log to be used in a number of different future methods. Although each individual
   log may start from the first index, this method will autmatically keep them in order.

   Usage:

   >>> complete_log = concat_training_logs(history, history2.history, "history3.csv")

   Arguments:
      - logs: A single or multiple training history objects, or the path to a csv containing the
              training history from a training session.
      - save: Whether you want to save the log to a csv file. If you do, then input the
              csv path as the value for this argument.
   Returns:
      - The DataFrame object containing the training logs.
   """
   final_log = logs[0]
   for log in logs[1:]:
      try:
         # Get the final index of the past log to continue from, and
         # then update the current log item with the proper indexes.
         final_index = next(final_log.iloc[[-1]].iterrows())[0]
         log.index = pd.RangeIndex(final_index + 1, final_index + len(log) + 1)

         # Concatenate current log with new log item.
         final_log = pd.concat([final_log, log])

      except Exception as e:
         raise e

   # Save log if it states to save.
   if save:
      try:
         final_log.to_csv(save)
      except Exception as e:
         if not isinstance(save, str):
            raise ValueError("If you want to save the training log, you need to provide a save "
                             "path for the `save` argument.")
         else:
            raise e

   # Return the final training log.
   return final_log


