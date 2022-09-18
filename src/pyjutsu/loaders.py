"""
Helpers for dynamic classes,modules,whatever loading.
"""

import importlib


def load_class(module_path, name):
    components = name.split('.')
    c = importlib.import_module(module_path)
    for comp in components:
        c = getattr(c, comp)
    return c


def load_module(module_uri):
    return __import__(module_uri, fromlist=[''])