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

from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.types import SupervisedModel
from pydvl.valuation.types import IndexT, Sample, SampleT
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["DataUtilityLearning"]


class DataUtilityLearning(UtilityBase[SampleT]):
    """This object wraps a [Utility][pydvl.valuation.utility.Utility] and delegates
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
        ``` pycon
        >>> from pydvl.valuation.dataset import Dataset
        >>> from pydvl.valuation.utility import ModelUtility, DataUtilityLearning
        >>> from pydvl.valuation.types import Sample
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>>
        >>> train, test = Dataset.from_sklearn(load_iris())
        >>> u = ModelUtility(LogisticRegression())
        >>> wrapped_u = DataUtilityLearning(u.with_dataset(train), 3, LinearRegression())
        ... # First 3 calls will be computed normally
        >>> for i in range(3):
        ...     _ = wrapped_u(Sample(0, np.array([])))
        >>> wrapped_u(Sample(0, np.array([1, 2, 3]))) # Subsequent calls will be computed using the fit model for DUL
        0.0
        ```

    """

    def __init__(
        self, utility: UtilityBase, training_budget: int, model: SupervisedModel
    ) -> None:
        self.utility = utility
        self.training_budget = training_budget
        self.model = model
        self._current_iteration = 0
        self._is_fitted = False
        self._utility_samples: Dict[Sample, Tuple[NDArray[np.bool_], float]] = {}

    def _convert_indices_to_boolean_vector(
        self, x: Iterable[IndexT]
    ) -> NDArray[np.bool_]:
        assert self.utility.training_data is not None
        boolean_vector: NDArray[np.bool_] = np.zeros(
            (1, len(self.utility.training_data)), dtype=np.bool_
        )
        if x is not None:
            boolean_vector[:, tuple(x)] = True
        return boolean_vector

    def __call__(self, sample: Sample | None) -> float:
        if self.training_data is None:
            raise ValueError("No training data set for utility")
        if sample is None or len(sample.subset) == 0:
            return self.utility(None)

        indices_boolean_vector = self._convert_indices_to_boolean_vector(sample.subset)
        if len(self._utility_samples) < self.training_budget:
            utility = self.utility(sample)
            self._utility_samples[sample] = (indices_boolean_vector, utility)
        else:
            if not self._is_fitted:
                X_, y_ = zip(*self._utility_samples.values())
                X = np.vstack(X_)
                y = np.asarray(y_)
                self.model.fit(X, y)
                self._is_fitted = True
            if sample in self._utility_samples:
                utility = self._utility_samples[sample][1]
            else:
                utility = self.model.predict(indices_boolean_vector).item()
        return utility

    # Forward all other calls / property_accesses to the wrapped utility
    def __getattr__(self, item):
        return getattr(self.utility, item)

    def __setattr__(self, key, value):
        if key != "utility":
            setattr(self.utility, key, value)
        else:
            super().__setattr__(key, value)

    # Avoid infinite recursion in __getattr__ when pickling:
    #  my theory: pickle attempts to access .utility before it has been set, which
    #  triggers __getattr__ and the recursion
    # FIXME: test that we are really pickling correctly
    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
