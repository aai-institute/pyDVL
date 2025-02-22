"""
This module implements **Data Utility Learning** (Wang et al., 2022)<sup><a
href="#wang_improving_2022">1</a></sup>.

!!! fixme "Parallel processing not supported"
    As of 0.9.0, this method does not support parallel processing. DataUtilityLearning
    would have to collect all utility samples in a single process before fitting the
    model.

## References

[^1]: <a name="wang_improving_2022"></a>Wang, T., Yang, Y. and Jia, R., 2021.
    [Improving cooperative game theory-based data valuation via data utility
    learning](https://arxiv.org/abs/2107.06336). arXiv preprint arXiv:2107.06336.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Collection

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.types import SupervisedModel
from pydvl.valuation.types import Sample, SampleT
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["DataUtilityLearning", "IndicatorUtilityModel", "UtilityModel"]


logger = logging.getLogger(__name__)


class UtilityModel(ABC):
    """Interface for utility models.

    * fitted on dictionaries of Sample -> utility value
    * Predict [samples] -> [utility]

    """

    @abstractmethod
    def fit(self, x: dict[Sample, float]): ...

    @abstractmethod
    def predict(self, x: Collection[Sample]) -> NDArray[np.float64]: ...


class IndicatorUtilityModel(UtilityModel):
    """A simple wrapper for arbitrary predictors.

    Uses 1-hot encoding of the data as input for the model, as done in Wang et al.,
    (2022)<sup><a href="#wang_improving_2022">1</a></sup>.
    """

    def __init__(self, predictor: SupervisedModel, n_data: int):
        self.n_data = n_data
        self.predictor = predictor

    def fit(self, samples: dict[Sample, float]):
        n_samples = len(samples)
        x = np.zeros((n_samples, self.n_data))
        y = np.zeros((n_samples, 1))
        for i, (s, u) in enumerate(samples.items()):
            x[i, s.subset] = 1.0
            y[i] = u
        logger.info(f"Fitting utility model with {n_samples} samples")
        self.predictor.fit(x, y)

    def predict(self, x: Collection[Sample]) -> NDArray[np.float64]:
        mask = np.zeros((len(x), self.n_data))
        for i, s in enumerate(x):
            mask[i, s.subset] = 1.0
        return self.predictor.predict(mask)


class DataUtilityLearning(UtilityBase[SampleT]):
    """This object wraps any [utility][pydvl.valuation.utility] and delegates
    calls to it, up until a given budget (number of iterations). Every tuple
    of input and output (a so-called *utility sample*) is stored. Once the
    budget is exhausted, `DataUtilityLearning` fits the given model to the
    utility samples. Subsequent calls will use the learned model to predict the
    utility instead of delegating.

    Args:
        utility: The [Utility][pydvl.valuation.utility.Utility] to learn.
        training_budget: Number of utility samples to collect before fitting
            the given model.
        model: A supervised regression model

    ??? Example
        ``` python
        from pydvl.valuation import Dataset, DataUtilityLearning, ModelUtility, \
            Sample, SupervisedScorer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.datasets import load_iris

        train, test = Dataset.from_sklearn(load_iris())
        scorer = SupervisedScorer("accuracy", test, 0, (0,1))
        utility = ModelUtility(LinearRegression(), scorer)
        utility_model = IndicatorUtilityModel(LinearRegression(), len(train))
        dul = DataUtilityLearning(utility, 3, utility_model)
        # First 3 calls will be computed normally
        for i in range(3):
            _ = dul(Sample(0, np.array([])))
        # Subsequent calls will be computed using the fitted utility_model
        dul(Sample(0, np.array([1, 2, 3])))
        ```

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
        self, utility: UtilityBase, training_budget: int, model: UtilityModel
    ) -> None:
        self.utility = utility
        self.training_budget = training_budget
        self.model = model
        self.n_predictions = 0
        self._is_fitted = False
        self._utility_samples: dict[Sample, float] = {}

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
