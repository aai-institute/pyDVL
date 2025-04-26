"""
Scorers are a fundamental building block of many data valuation methods. They
are typically used by [ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility]
and its subclasses to evaluate the quality of a model when trained on subsets of the
training data.

Scorers evaluate trained models in user-defined ways, and provide additional
information about themselves, like their range and default value, which can be used by
some data valuation methods (e.g. [Group Testing
Shapley][pydvl.valuation.methods.gt_shapley]) to estimate the number of samples required
for a certain quality of approximation.
"""

from .base import *
from .classwise import *
from .supervised import *
from .torchscorer import *
from .utils import *
