#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import re
from collections import defaultdict

from deeptoolkit.internal.helpers import DefaultDictionary

LOG_DICT_NAMES = {
   'acc': 'Training Accuracy',
   'accuracy': 'Training Accuracy',
   'val_acc': 'Validation Accuracy',
   'val_accuracy': 'Validation Accuracy',
   'loss': 'Training Loss',
   'val_loss': 'Validation Loss'
}

ACTIVATION_FUNCTION_NAMES = {
   'relu': 'ReLU',
   'leakyrelu': 'LeakyReLU',
   'sigmoid': 'Sigmoid',
   'softmax': 'Softmax'
}

VALID_CONVERSION_LAYER_NAMES = (
   'conv2d', 'batchnormalization', 'upsampling2d',
   'linear', 'dropout', 'input', 'depthwiseconv2d',
   'separable_conv2d', 'maxpooling2d', 'averagepooling2d',
   'flatten', 'activation', 'dense', 'relu', 'softmax'
)

LAYER_TORCH_CONVERSIONS = DefaultDictionary()
LAYER_TORCH_CONVERSIONS['Separable_conv2d'] = 'SEPARABLE_CONV'
LAYER_TORCH_CONVERSIONS['Depthwise_conv2d'] = 'DEPTHWISE_CONV'
LAYER_TORCH_CONVERSIONS['Dense'] = 'Linear'
LAYER_TORCH_CONVERSIONS['Batchnormalization'] = 'BatchNorm2d'


