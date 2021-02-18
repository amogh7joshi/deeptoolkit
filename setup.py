#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import setuptools
from setuptools import setup
from setuptools.command.install import install

from deeptoolkit.internal.acquisition import load_dnn_files

# Validate Python Version (must be >= 3.7 for compatibility).
if sys.version_info[:2] < (3, 7):
   raise RuntimeError("In order to use DeepToolKit, Python version >= 3.7 is required.")

def get_long_description():
   """Get the long library description from package README."""
   with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as long_file:
      return long_file.read()

class PostInstallResourceAcquisition(install):
   """Run the resource acquisition script after installation (not working currently)."""
   def run(self):
      install.run(self)
      load_dnn_files(override = True)

# Setup library.
setup(
   name = 'deeptoolkit',
   version = '0.2.0',
   author = 'Amogh Joshi',
   author_email = 'joshi.amoghn@gmail.com',
   description = 'A deep learning library containing implementations of popular algorithms and extensions to TensorFlow and Keras.',
   long_description = get_long_description(),
   long_description_content_type = 'text/markdown',
   url = 'https://github.com/amogh7joshi/deeptoolkit',
   classifiers = [
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      'Operating System :: OS Independent',
   ],
   cmdclass = {
     'install': PostInstallResourceAcquisition
   },
   packages = setuptools.find_packages(),
   include_package_data = True
)

