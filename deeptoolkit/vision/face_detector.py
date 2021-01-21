#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import pkg_resources
from typing import Any

import cv2
import numpy as np

__all__ = ['FacialDetector']

class FacialDetector(object):
   """Facial Detector object, using DNN for facial detection.

   This class contains a DCNN trained to detect faces, and can be used as the base
   facial detector in systems and processes involving facial detection. The model is
   based off of ResNet, and the original training script can be found in the OpenCV
   repository at https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector.
   The class can be used to detect faces from a live video or images.

   Usage:

   The model can be loaded directly from the program, and used as would be any other neural network.

   >>> model = FacialDetector()
   >>> image = cv2.imread('image/file/path')
   >>> faces = detect_face(image)

   For more information on the `detect_face` method, see below.

   Arguments:
      - directory: An optional argument, if you want to use your own architecture + weights files.
                   They default to the ResNet model from the OpenCV repository.
   """
   def __init__(self, directory = None):
      # Acquire resources and initialize data if possible.
      self._initialize_resources(directory)

      # Load DNN model.
      try:
         self.net = cv2.dnn.readNetFromCaffe(self.dnn_architecture, self.dnn_weights)
      except Exception as e:
         raise e

   def _initialize_resources(self, directory = None):
      """Initialize the facial detector resources (architecture + weights files)."""
      if directory: # If a directory path is provided, validate and load files.
         directory_files = os.listdir(directory)

         # Verify and set architecture file.
         arch_files = [file for file in directory_files if file.endswith('txt')]
         if len(arch_files) != 1:
            raise ValueError(f"Expected a single .txt or .prototxt architecture file for DNN model architecture, "
                             f"but got {len(arch_files)} files: {arch_files}.")
         else:
            self.dnn_architecture = arch_files[0]

         # Verify and set weights file.
         weights_files = [file for file in directory_files if file.endswith('.caffemodel')]
         if len(weights_files) != 1:
            raise ValueError(f"Expected a single .caffemodel weights file for DNN model weights, but got "
                             f"{len(weights_files)} files: {weights_files}.")
         else:
            self.dnn_weights = weights_files[0]
      else: # Otherwise, load default package files.
         self.dnn_architecture = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              'internal', 'resources', 'model.prototxt')
         self.dnn_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         'internal', 'resources', 'dnn-weights.caffemodel')
         if not os.path.exists(self.dnn_architecture) or not os.path.exists(self.dnn_weights):
            raise FileNotFoundError("DNN architecture + weights resource files not found, you will need to "
                                    "load them into the library again. Call deeptoolkit.load_dnn_files().")

   def detect_face(self, image, *, save: Any = False):
      """Primary method to detect a face from an image.

      Detects faces from an image and draws bounding boxes around them. Images can be
      saved to a file by updating the `save` parameter, and the method returns the annotated image.

      Usage:

      >>> model = FacialDetector()
      >>> image = model.detect_face(cv2.imread('image/file/path'), save = 'image/save/path')

      Arguments:
         - image: A numpy array representing the image, or an image file path.
         - save: Either a filepath to save the annotated image to or None.
      Returns:
         - The annotated image, with bounding boxes around faces.
      """
      # Validate image.
      if not isinstance(image, np.ndarray):
         if isinstance(image, str):
            if not os.path.exists(image):
               raise FileNotFoundError(f"Received image path string, but the path {image} does not exist.")
            image = cv2.imread(image)
         else:
            raise TypeError(f"Expected a numpy array representing the image, got {type(image)}.")
      if len(image.shape) != 3:
         raise ValueError(f"Image should have three dimensions: width, height, and channels, "
                          f"got {len(image.shape)} dims.")

      # Detect faces from image.
      image_coords = []
      (h, w) = image.shape[:2]

      # Set input blob and forward pass through network.
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), swapRB = False, crop = False)
      self.net.setInput(blob)
      faces = self.net.forward()

      # Iterate over faces.
      for dim in range(0, faces.shape[2]):
         # Determine prediction confidence.
         confidence = faces[0, 0, dim, 2]
         if confidence < 0.5:
            continue

         # Determine actual face coordinates.
         box = faces[0, 0, dim, 3:7] * np.array([w, h, w, h])
         (x, y, xe, ye) = box.astype(int)
         image_coords.append((x, y, xe, ye))

         # Annotate image with bounding box.
         cv2.rectangle(image, (x, y), (xe, ye), (0, 255, 255), 3)

      # Save image if requested to.
      if save:
         if not isinstance(save, str):
            raise TypeError(f"The save argument should be a path where the image is going "
                            f"to be saved, got {type(save)}.")
         if not os.path.exists(os.path.dirname(save)):
            raise NotADirectoryError("The directory of the save path provided does not exist. Check your paths.")
         cv2.imwrite(save, image)

      # Return annotated image.
      return image


