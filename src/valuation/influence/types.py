"""
Contains enumerations for influence functions.
"""
from enum import Enum


class InfluenceTypes(Enum):
    """
    Different influence types.
    """

    Up = 1
    Perturbation = 2
