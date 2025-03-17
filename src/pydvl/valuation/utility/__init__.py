"""
This module contains classes to manage and learn utility functions for the
computation of values.

[ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility] holds information
about model, data and scoring function (the latter being what one usually understands
under *utility* in the general definition of Shapley value). Model-based
evaluation methods define the utility as a retraining of the model on a subset
of the data, which is then [scored][pydvl.valuation.scorers]. Please see the
documentation on [Computing Data Values][computing-data-values] for more
information.

Utilities can be automatically cached across machines when the
[cache is so configured][getting-started-cache] and enabled upon construction.

[DataUtilityLearning][pydvl.valuation.utility.learning.DataUtilityLearning] adds support
for learning the scoring function to avoid repeated re-training of the model to
compute the score. Several methods exist to learn the utility function.
"""

from .modelutility import *  # isort: skip
from .classwise import *
from .knn import *
from .learning import *

try:
    from .torch import *
except ImportError:
    pass
