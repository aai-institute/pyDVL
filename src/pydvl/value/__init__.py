r"""
Algorithms for the exact and approximate computation of value and semi-value.

See :ref:`data valuation` for an introduction to the concepts and methods
implemented here.
"""

from .shapley import ShapleyMode, compute_shapley_values
from .valuationresult import ValuationResult
