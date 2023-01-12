# coding=utf-8
"""
  @file: loader.py
  @data: 28 September 2022
  @author: cecabert

  Loader utility
"""
import os
import sys
from importlib import import_module


def _is_py_file(x: str) -> bool:
    return '.py' in x


def load_module(module_or_filename: str):
    m = module_or_filename
    if _is_py_file(m):
        # Add folder to `sys.path`
        folder = os.path.dirname(m)
        m, _ = os.path.splitext(os.path.basename(m))
        if folder:
            sys.path.append(folder)
    # Load module
    module = import_module(m)
    return module
