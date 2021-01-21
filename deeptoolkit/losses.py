#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import

from tensorflow.keras.losses import Loss

import deeptoolkit.core.functional as F
from deeptoolkit.internal.conversion import apply_tensor_conversion

__all__ = ['BinaryFocalLoss', 'CategoricalFocalLoss']

class BinaryFocalLoss(Loss):
   """Binary focal loss function from the paper `Focal Loss for Dense Object Detection`.

   Computes binary focal loss given a ground truth object and a prediction object.

   Usage:

   You can use it to directly compute loss on two inputted object.

   >>> bfl = BinaryFocalLoss()
   >>> truth = [1, 2, 3, 4]
   >>> predicted = [0.97, 2.6, 4.1, 2.9]
   >>> loss_value = bfl(truth, predicted)

   You can also use it in a regular Keras model training pipeline.

   >>> model.compile(
   >>>   optimizer = 'adam',
   >>>   loss = BinaryFocalLoss(),
   >>>   metrics = ['accuracy']
   >>> )

   Arguments:
      - y_true: A ground truth object.
      - y_pred: A prediction object.
      - gamma: Focus parameter for the modulating factor.
      - alpha: The weighting factor in cross-entropy.
   Returns:
      - A 1-dimensional Tensor containing the loss value.
   """
   def __init__(self, *, gamma = 2.0, alpha = 0.25):
      super(BinaryFocalLoss, self).__init__()
      self.gamma = gamma
      self.alpha = alpha

   @apply_tensor_conversion
   def call(self, y_true, y_pred, gamma = 2.0, alpha = 0.25):
      return F.binary_focal_loss(y_true, y_pred, gamma = gamma, alpha = alpha)

class CategoricalFocalLoss(Loss):
   """Categorical focal loss function from the paper `Focal Loss for Dense Object Detection.`

   Computes binary focal loss given a ground truth object and a prediction object.

   Usage:

   You can use it to directly compute loss on two inputted object.

   >>> cfl = CategoricalFocalLoss()
   >>> truth = [1, 2, 3, 4]
   >>> predicted = [0.86, 2.1, 4.9, 3.6]
   >>> loss_value = cfl(truth, predicted)

   You can also use it in a regular Keras model training pipeline.

   >>> model.compile(
   >>>   optimizer = 'adam',
   >>>   loss = CategoricalFocalLoss(),
   >>>   metrics = ['accuracy']
   >>> )

   Arguments:
      - y_true: A ground truth object.
      - y_pred: A prediction object.
      - gamma: Focus parameter for the modulating factor.
      - alpha: The weighting factor in cross-entropy.
   Returns:
      - A 1-dimensional Tensor containing the loss value.
   """
   def __init__(self, *, gamma = 2.0, alpha = 0.25):
      super(CategoricalFocalLoss, self).__init__()
      self.gamma = gamma
      self.alpha = alpha

   @apply_tensor_conversion
   def call(self, y_true, y_pred):
      return F.categorical_focal_loss(y_true, y_pred, gamma = self.gamma, alpha = self.alpha)

