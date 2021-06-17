#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

def in_interactive_environment():
   """Determines whether the script is running in an interactive shell."""
   return "get_ipython" in globals()

