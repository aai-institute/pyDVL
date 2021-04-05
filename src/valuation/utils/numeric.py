import numpy as np

from functools import lru_cache
from itertools import chain, combinations
from typing import Iterator, Iterable
from valuation.utils.dataset import Dataset
from valuation.utils.types import SupervisedModel


def vanishing_derivatives(values: np.ndarray,
                          min_values: int,
                          value_tolerance: float):
    """Checks empirical convergence of the derivatives of rows to zero
    and returns the number of rows that have converged. """
    last_values = values[:, -min_values - 1:]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=value_tolerance).sum(axis=1)
    return np.sum(zeros >= min_values / 2)


def powerset(it: Iterable) -> Iterator:
    """ Returns an iterator for the power set of
    >>> powerset([1,2])
    () (1,) (2,) (1,2)
    """
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# There is a clever way of rearranging the loops to fit just once per
# list of indices. Or... we can just cache and be done with it.
# FIXME: make usage of the cache optional for cases where it is not necessary
# TODO: benchmark this
@lru_cache
def utility(model: SupervisedModel,
            data: Dataset,
            indices: Iterable[int],
            catch_errors: bool = True) -> float:
    """ Fits the model on a subset of the training data and scores it on the
    test data.
    :param model: Any supervised model
    :param data: a split Dataset
    :param indices: a subset of indices from data.x_train.index
    :param catch_errors: set to True to return np.nan if fit() fails. This hack
        helps when a step in a pipeline fails if there are too few data points
    :return: 0 if no indices are passed, otherwise the value of model.score on
        the test data.
    """
    if not indices:
        return 0.0
    x = data.x_train.iloc[list(indices)]
    y = data.y_train.iloc[list(indices)]
    try:
        model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
        return model.score(data.x_test, data.y_test)
    except Exception as e:
        if catch_errors:
            return np.nan
        else:
            raise e


def lower_bound_hoeffding(delta: float, eps: float, r: float) -> int:
    """ Minimum number of samples required for MonteCarlo Shapley to obtain
    an (eps,delta) approximation.
    That is, with probability 1-delta, the estimate will be epsilon close to
     the true quantity, if at least so many monte carlo samples are taken.
    """
    return int(np.ceil(np.log(2 / delta) * r ** 2 / (2 * eps ** 2)))
