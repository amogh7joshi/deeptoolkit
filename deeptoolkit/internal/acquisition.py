#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import subprocess
import logging

def load_dnn_files(override = False):
   """Acquire the DNN resource files (architecture + weights) from their respective sources.

   This function calls upon the script get-resources.sh to get the default ResNet DNN architecture
   files from the OpenCV repository, for usage in the FacialDetector object. The files are then
   acquired and placed in the deeptoolkit/internal/resources directory, to be used internally.

   Usage:

   You can place the method at the top of a script, and with the `override` argument set to False,
   the method will automatically skip with each iteration if the files already exist.

   >>> from deeptoolkit import load_dnn_files
   >>> load_dnn_files(override = False)

   """
   # Determine whether files already exist.
   resource_path = os.path.join(os.path.dirname(__file__), 'resources')
   if os.path.exists(os.path.join(resource_path, 'model.prototxt')) and \
      os.path.exists(os.path.join(resource_path, 'dnn-weights.caffemodel')):
      if not override:
         return
      else: # Warn the user beforehand if files already exist, so they know to hit cancel if necessary.
         logging.warning("The DNN model files already exist and you are overwriting them. "
                         "If you want to retain existing files, cancel the script at the prompt.")

   # Execute script.
   script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'get-resources.sh')
   subprocess.run(['bash', script_path, resource_path])
