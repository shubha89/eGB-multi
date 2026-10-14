"""Compatibility alias for the canonical egb_multi.eccentric implementation."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("egb_multi.eccentric")
