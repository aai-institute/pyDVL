"""
This module implements **Data Utility Learning** (Wang et al., 2022).[^1]

DUL uses an ML model to learn the utility function. Essentially, it learns to predict
the performance of a model when trained on a given set of indices from the dataset. The
cost of training this model is quickly amortized by avoiding costly re-evaluations of
the original utility.

Usage is through the [DataUtilityLearning] class, which wraps any utility function and
a [UtilityModel][pydvl.valuation.utility.learning.UtilityModel] to learn it. The
wrapper collects utility samples until a given budget is reached, and then fits
the model. After that, it forwards any queries for utility values to this learned model
to predict the utility of new samples at constant, and low, cost.

See [the documentation][data-utility-learning-intro] for more information.

!!! todo
    DUL does not support parallel training of the model yet. This is a limitation of the
    current architecture. Additionally, batching of utility evaluations should be added
    to really profit from neural network architectures.

## References

[^1]: <a name="wang_improving_2022"></a>Wang, T., Yang, Y. and Jia, R., 2021.
    [Improving cooperative game theory-based data valuation via data utility
    learning](https://arxiv.org/abs/2107.06336). arXiv preprint arXiv:2107.06336.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Collection, Generic

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pydvl.utils.array import ArrayRetT
from pydvl.utils.functional import suppress_warnings
from pydvl.valuation.types import Sample, SampleT, SupervisedModel
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["DataUtilityLearning", "IndicatorUtilityModel", "UtilityModel"]


logger = logging.getLogger(__name__)


class UtilityModel(ABC, Generic[ArrayRetT]):
    """Interface for utility models.

    A _utility model_ predicts the value of a utility function given a sample. The model
    is trained on a collection of samples and their respective utility values. These
    tuples are called _Utility Samples_.

    Utility models:

    * are fitted on dictionaries of Sample -> utility value
    * predict: Collection[samples] -> Array[utility values]
    """

    @abstractmethod
    def fit(self, x: dict[Sample, float]) -> Self: ...

    @abstractmethod
    def predict(self, x: Collection[Sample]) -> ArrayRetT: ...


class IndicatorUtilityModel(UtilityModel[NDArray]):
    """A simple wrapper for arbitrary predictors.

    Uses 1-hot encoding of the indices as input for the model, as done in Wang et al.,
    (2022)<sup><a href="#wang_improving_2022">1</a></sup>.

    This encoding can be fed to any regressor. See [the
    documentation][dul-indicator-encoding-intro] for details.

    Args:
        predictor: A supervised model that implements the `fit` and `predict` methods.
            This model will be trained on the encoded utility samples gathered by the
            [DataUtilityLearning][pydvl.valuation.utility.learning.DataUtilityLearning]
            object.
        n_data: Number of indices in the dataset. This is used to create the input
            matrix for the model.
    """

    def __init__(self, predictor: SupervisedModel[NDArray, NDArray], n_data: int):
        self.n_data = n_data
        self.predictor = predictor

    def fit(self, samples: dict[Sample, float]) -> Self:
        n_samples = len(samples)
        x = np.zeros((n_samples, self.n_data))
        y = np.zeros((n_samples, 1))
        for i, (s, u) in enumerate(samples.items()):
            x[i, s.subset] = 1.0
            y[i] = u
        logger.info(f"Fitting utility model with {n_samples} samples")
        self.predictor.fit(x, y)
        return self

    def predict(self, x: Collection[Sample]) -> NDArray:
        mask = np.zeros((len(x), self.n_data))
        for i, s in enumerate(x):
            mask[i, s.subset] = 1.0
        return self.predictor.predict(mask)


class DataUtilityLearning(UtilityBase[SampleT]):
    """This object wraps any class derived from
    [UtilityBase][pydvl.valuation.utility.base.UtilityBase] and delegates calls to it,
    up until a given budget (number of iterations). Every tuple of input and output (a
    so-called *utility sample*) is stored. Once the budget is exhausted,
    `DataUtilityLearning` fits the given model to the utility samples. Subsequent
    calls will use the learned model to predict the utility instead of delegating.

    Args:
        utility: The utility to learn. Typically, this will be a
            [ModelUtility][pydvl.valuation.utility.ModelUtility] object encapsulating
            a machine learning model which requires fitting on each evaluation of the
            utility.
        training_budget: Number of utility samples to collect before fitting the given
            model.
        model: A wrapper for a supervised model that can be trained on a collection of
            utility samples.
    """

    # Attributes that belong to this proxy. All other attributes are forwarded to the
    # wrapped utility.
    _local_attrs = {
        "utility",
        "training_budget",
        "model",
        "_current_iteration",
        "_is_fitted",
        "_utility_samples",
        "_n_predictions",
    }

    def __init__(
        self,
        utility: UtilityBase,
        training_budget: int,
        model: UtilityModel,
        show_warnings: bool = True,
    ) -> None:
        self.utility = utility
        self.training_budget = training_budget
        self.model = model
        self.n_predictions = 0
        self.show_warnings = show_warnings
        self._is_fitted = False
        self._utility_samples: dict[Sample, float] = {}

    @suppress_warnings(flag="show_warnings")
    def __call__(self, sample: Sample | None) -> float:
        if self.training_data is None:
            raise ValueError("No training data set for utility")

        if sample is None or len(sample.subset) == 0:
            return self.utility(sample)

        if sample in self._utility_samples:
            return self._utility_samples[sample]

        if len(self._utility_samples) < self.training_budget:
            utility = self.utility(sample)
            self._utility_samples[sample] = utility
            return utility

        if not self._is_fitted:
            self.model.fit(self._utility_samples)
            self._is_fitted = True

        self.n_predictions += 1
        return float(self.model.predict([sample]).item())

        # Strictly speaking, (sample, prediction) is not a utility sample, and it's
        # unlikely that we will ever hit the same sample twice, so we don't store it:
        # self._utility_samples[sample] = utility

    # Forward all other calls / property_accesses to the wrapped utility
    def __getattr__(self, item):
        return getattr(self.utility, item)

    def __setattr__(self, key, value):
        if key in self._local_attrs:
            object.__setattr__(self, key, value)
        else:
            setattr(self.utility, key, value)

    # Avoid infinite recursion in __getattr__ when pickling:
    #  my theory: pickle attempts to access .utility before it has been set, which
    #  triggers __getattr__ and the recursion
    # FIXME: test that we are really pickling correctly
    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
