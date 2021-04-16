import numpy as np

from functools import lru_cache
from itertools import chain, combinations
from random import getrandbits
from typing import Generator, Iterator, Iterable, List, TypeVar
from sklearn.metrics import check_scoring

from valuation.utils.dataset import Dataset
from valuation.utils.types import Scorer, SupervisedModel

T = TypeVar('T')


def vanishing_derivatives(x: np.ndarray, min_values: int, eps: float) -> int:
    """ Returns the number of rows whose empirical derivatives have converged
        to zero, up to a tolerance of eps.
    """
    last_values = x[:, -min_values - 1:]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=eps).sum(axis=1)
    return int(np.sum(zeros >= min_values / 2))


def powerset(it: Iterable[T]) -> Iterator[Iterable[T]]:
    """ Returns an iterator for the power set of the argument.

    Subsets are generated in sequence by growing size. See `random_powerset()`
    for random sampling.

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
            catch_errors: bool = True,
            scoring: Scorer = None)\
        -> float:
    """ Fits the model on a subset of the training data and scores it on the
    test data. Results are memoized to avoid duplicate computation.

    :param model: Any supervised model
    :param data: a split Dataset
    :param indices: a subset of indices from data.x_train.index. The type must
      be hashable for the caching to work, e.g. wrap the argument with
      `frozenset()` (rather than `tuple()` since order should not matter)
    :param catch_errors: set to True to return np.nan if fit() fails. This hack
        helps when a step in a pipeline fails if there are too few data points
    :param scoring: Same as in sklearn's `cross_validate()`: a string, a scorer
        callable or None for the default `model.score()`.
    :return: 0 if no indices are passed, otherwise the value of model.score on
        the test data.
    """
    if not indices:
        return 0.0
    scorer = check_scoring(model, scoring)
    x = data.x_train.iloc[list(indices)]
    y = data.y_train.iloc[list(indices)]
    try:
        model.fit(x.values, y.values)
        return scorer(model, data.x_test, data.y_test)
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


def random_subset_indices(n: int) -> List[int]:
    """ Uniformly samples a subset of indices in the range [0,n).
    :param n: number of indices.
    """
    if n <= 0:
        return []
    r = getrandbits(n)
    indices = []
    for b in range(n):
        if r & 1:
            indices.append(b)
        r = r >> 1
    return indices


def random_powerset(indices: np.ndarray, max_subsets: int = None) \
        -> Generator[np.ndarray, None, None]:
    """ Uniformly samples a subset from the power set of the argument, without
    pre-generating all subsets and in no order.

    See `powerset()` if you wish to deterministically generate all subsets.
    :param indices:
    :param max_subsets: if set, stop the generator after this many steps.
    """
    n = len(indices)
    total = 1
    while True and total <= max_subsets:
        subset = random_subset_indices(n)
        yield indices[subset]
        total += 1


def symmetric_mean_absolute_percentage_error(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: ArrayLike = None,
        multioutput: Union[ArrayLike, Literal['raw_values', 'uniform_average']]
                        = 'uniform_average') -> Union[float, np.ndarray]:
    """ Symmetric mean absolute percentage error regression loss.

    Computes:

    $$1/n \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|} $$


    Parameters
    ----------
    y_true : shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight :shape (n_samples,), default=None
        Sample weights.
    multioutput :
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
        Scalar(s) in [0,1]. The best return value is 0.0, the worst 1.0
        If multioutput is 'raw_values', then sMAPE is returned for each output
        separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred)
    0.289393...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred)
    0.304040...
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.283434...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred,
                                                             multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    smape = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


smape_scorer = make_scorer(symmetric_mean_absolute_percentage_error,
                           greater_is_better=False)
