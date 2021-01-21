#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from deeptoolkit.internal.helpers import select_threshold

def binary_focal_loss(gt, pred, *, gamma = 2.0, alpha = 0.25):
   """Implementation of binary focal loss.

   This is the binary focal loss function from the paper on focal losses,
   `Focal Loss For Object Detection`: https://arxiv.org/abs/1708.02002.

   Formula: Focal Loss * Cross Entropy

      J = -gt * alpha * ((1-pred) ^ gamma) * log(pred)
          - (1 - gt) * alpha * (pred ^ gamma) * log(1 - pred)

   Arguments:
      - gt: Ground Truth (y_true)
      - pred: Prediction (y_pred)
      - gamma: Focus parameter for the modulating factor.
      - alpha: The weighting factor in cross-entropy.
   """
   # Cast and clip truth/predictions (to prevent errors).
   if not gt.dtype == tf.float32:
      gt = tf.cast(gt, tf.float32)
   if not pred.dtype == tf.float32:
      pred = tf.cast(pred, tf.float32)
   pred = K.clip(pred, K.epsilon(), 1.0 - K.epsilon())

   # Calculate cross-entropy and focal losses.
   cross_entropy = -gt * (alpha * K.pow(1 - pred, gamma) * K.log(pred))
   focal_loss = -(1 - gt) * ((1 - alpha) * K.pow(pred, gamma) * K.log(1 - pred))

   # Calculate and return final loss.
   loss = K.mean(cross_entropy + focal_loss)
   return loss

def categorical_focal_loss(gt, pred, *, gamma = 2.0, alpha = 0.25):
   """Implementation of categorical focal loss.

   This is the categorical focal loss function from the paper on focal losses,
   `Focal Loss For Object Detection`: https://arxiv.org/abs/1708.02002.

   Formula: Focal Loss

      loss = -gt * alpha * ((1 - pred) ^ gamma) * log(pred)

   Arguments:
      - gt: Ground Truth (y_true)
      - pred: Prediction (y_pred)
      - gamma: Focus parameter for the modulating factor.
      - alpha: The weighting factor in cross-entropy.
   """
   # Cast and clip truth/predictions (to prevent errors).
   if not gt.dtype == tf.float32:
      gt = tf.cast(gt, tf.float32)
   if not pred.dtype == tf.float32:
      pred = tf.cast(pred, tf.float32)
   pred = K.clip(pred, K.epsilon(), 1.0 - K.epsilon())

   # Calculate focal loss.
   focal_loss = -gt * (alpha * K.pow(1 - pred, gamma) * K.log(pred))

   # Calculate and return final loss.
   loss = K.mean(focal_loss)
   return loss

