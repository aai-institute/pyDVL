"""
This module contains classes to manage and learn utility functions for the
computation of values. Please see the documentation on
[Computing Data Values][computing-data-values] for more information.

[Utility][pydvl.valuation.utility.Utility] holds information about model,
data and scoring function (the latter being what one usually understands
under *utility* in the general definition of Shapley value).
It is automatically cached across machines when the
[cache is configured][getting-started-cache] and it is enabled upon construction.

[DataUtilityLearning][pydvl.valuation.utility.DataUtilityLearning] adds support
for learning the scoring function to avoid repeated re-training
of the model to compute the score.
"""
from .learning import *
from .utility import *
