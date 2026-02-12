"""
VocalBaby - Production ML System for Infant Cry Analysis

A production-grade audio classification system for analyzing infant cries
using XGBoost models and eGeMAPS acoustic features.
"""

__version__ = "1.0.0"
__author__ = "Arun Prakash Singh"

# Key exports
from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging

__all__ = [
    "__version__",
    "__author__",
    "VocalBabyException",
    "logging",
]
