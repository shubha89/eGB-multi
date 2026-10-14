"""Compatibility alias for the referee-validated instrument-noise helpers."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("egb_multi.noise")
