"""
This module provides a :class:`Scorer` class that wraps scoring functions with
additional information.

Scorers can be constructed in the same way as in scikit-learn: either from
known strings or from a callable. Greater values must be better. If they are not,
a negated version can be used, see scikit-learn's `make_scorer()
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`_.

:class:`Scorer` provides additional information about the scoring function, like
its range and default values, which can be used by some data valuation
methods (like :func:`~pydvl.value.shapley.gt.group_testing_shapley`) to estimate
the number of samples required for a certain quality of approximation.
"""
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.metrics import accuracy_score, get_scorer, make_scorer

from pydvl.utils.types import SupervisedModel

__all__ = [
    "Scorer",
    "ClasswiseScorer",
    "compose_score",
    "squashed_r2",
    "squashed_variance",
]


class ScorerCallable(Protocol):
    """Signature for a scorer"""

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        ...


class Scorer:
    """A scoring callable that takes a model, data, and labels and returns a
    scalar.

    :param scoring: Either a string or callable that can be passed to
        `get_scorer
        <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html>`_.
    :param default: score to be used when a model cannot be fit, e.g. when too
        little data is passed, or errors arise.
    :param range: numerical range of the score function. Some Monte Carlo
        methods can use this to estimate the number of samples required for a
        certain quality of approximation. If not provided, it can be read from
        the ``scoring`` object if it provides it, for instance if it was
        constructed with :func:`~pydvl.utils.types.compose_score`.
    :param name: The name of the scorer. If not provided, the name of the
        function passed will be used.

    .. versionadded:: 0.5.0

    """

    _name: str
    range: NDArray[np.float_]

    def __init__(
        self,
        scoring: Union[str, ScorerCallable],
        default: float = 0.0,
        range: Tuple = (-np.inf, np.inf),
        name: Optional[str] = None,
    ):
        if name is None and isinstance(scoring, str):
            name = scoring
        self._scorer = get_scorer(scoring)
        self.default = default
        # TODO: auto-fill from known scorers ?
        self.range = np.array(range)
        self._name = getattr(self._scorer, "__name__", name or "scorer")

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return self._scorer(model, X, y)  # type: ignore

    def __str__(self):
        return self._name

    def __repr__(self):
        capitalized_name = "".join(s.capitalize() for s in self._name.split(" "))
        return f"{capitalized_name} (scorer={self._scorer})"


class ClasswiseScorer(Scorer):
    """A Scorer which is applicable for valuation in classification problems. Its value
    is based on in-cls and out-of-cls score :footcite:t:`schoch_csshapley_2022`. For
    each class ``label`` it separates the elements into two groups, namely in-cls
    instances and out-of-cls instances. The value function itself than estimates the
    in-cls metric discounted by the out-of-cls metric. In other words the value function
    for each element of one class is conditioned on the out-of-cls instances (or a
    subset of it). The form of the value function can be written as

    .. math::
        v_{y_i}(D) = f(a_S(D_{y_i}))) * g(a_S(D_{-y_i})))

    where f and g are continuous, monotonic functions and D is the test set.

    in order to produce meaningful results. For further reference see also section four
    of :footcite:t:`schoch_csshapley_2022`.

    :param default: Score used when a model cannot be fit, e.g. when too little data is
        passed, or errors arise.
    :param range: Numerical range of the score function. Some Monte Carlo methods can
        use this to estimate the number of samples required for a certain quality of
        approximation. If not provided, it can be read from the ``scoring`` object if it
        provides it, for instance if it was constructed with
        :func:`~pydvl.utils.types.compose_score`.
    :param in_class_discount_fn: Continuous, monotonic increasing function used to
        discount the in-class score.
    :param out_of_class_discount_fn: Continuous, monotonic increasing function used to
        discount the out-of-class score.
    :param initial_label: Set initial label (Doesn't require to set parameter ``label``
        on ``ClassWiseDiscountedScorer`` in first iteration)
    :param name: Name of the scorer. If not provided, the name of the passed
        function will be prefixed by 'classwise '.

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        scoring: str = "accuracy",
        default: float = 0.0,
        range: Tuple[float, float] = (-np.inf, np.inf),
        in_class_discount_fn: Callable[[float], float] = lambda x: x,
        out_of_class_discount_fn: Callable[[float], float] = np.exp,
        initial_label: Optional[int] = None,
        name: Optional[str] = None,
    ):
        disc_score_in_cls = in_class_discount_fn(range[1])
        disc_score_out_of_cls = out_of_class_discount_fn(range[1])
        transformed_range = (0, disc_score_in_cls * disc_score_out_of_cls)
        super().__init__(
            "accuracy",
            range=transformed_range,
            default=default,
            name=name or f"classwise {scoring}",
        )
        self._in_cls_discount_fn = in_class_discount_fn
        self._out_of_cls_discount_fn = out_of_class_discount_fn
        self.label = initial_label

    def __str__(self):
        return self._name

    def __call__(
        self: "ClasswiseScorer",
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
    ) -> float:
        """
        :param model: Model used for computing the score on the validation set.
        :param x_test: Array containing the features of the classification problem.
        :param y_test: Array containing the labels of the classification problem.
        :return: Calculated score.
        """
        in_cls_score, out_of_cls_score = self.estimate_in_cls_and_out_of_cls_score(
            model, x_test, y_test
        )
        disc_score_in_cls = self._in_cls_discount_fn(in_cls_score)
        disc_score_out_of_cls = self._out_of_cls_discount_fn(out_of_cls_score)
        return disc_score_in_cls * disc_score_out_of_cls

    def estimate_in_cls_and_out_of_cls_score(
        self,
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
        rescale_scores: bool = True,
    ) -> Tuple[float, float]:
        r"""
        Computes in-class and out-of-class scores using the provided scoring function,
        which can be expressed as:

        .. math::
            a_S(D=\{(\hat{x}_1, \hat{y}_1), \dots, (\hat{x}_K, \hat{y}_K)\}) &=
            \frac{1}{N} \sum_k s(y(\hat{x}_k), \hat{y}_k)

        In this context, the computation is performed twice: once on D_i and once on D_o
        to calculate the in-class and out-of-class scores. Here, D_i contains only
        samples with the specified 'label' from the validation set, while D_o contains
        all other samples. By default, the scores are scaled to have the same order of
        magnitude. In such cases, the raw scores are multiplied by:

        .. math::
            N_{y_i} = \frac{a_S(D_{y_i})}{a_S(D_{y_i})+a_S(D_{-y_i})} \quad \text{and}
            \quad N_{-y_i} = \frac{a_S(D_{-y_i})}{a_S(D_{y_i})+a_S(D_{-y_i})}

        :param model: Model used for computing the score on the validation set.
        :param x_test: Array containing the features of the classification problem.
        :param y_test: Array containing the labels of the classification problem.
        :param rescale_scores: If set to True, the scores will be denormalized. This is
            particularly useful when the inner score is calculated by an estimator of
            the form 1/N sum_i x_i.
        :return: Tuple containing the in-class and out-of-class scores.
        """
        scorer = self._scorer
        label_set_match = y_test == self.label
        label_set = np.where(label_set_match)[0]
        num_classes = len(np.unique(y_test))

        if len(label_set) == 0:
            return 0, 1 / (num_classes - 1)

        complement_label_set = np.where(~label_set_match)[0]
        in_cls_score = scorer(model, x_test[label_set], y_test[label_set])
        out_of_cls_score = scorer(
            model, x_test[complement_label_set], y_test[complement_label_set]
        )

        if rescale_scores:
            n_in_cls = np.count_nonzero(y_test == self.label)
            n_out_of_cls = len(y_test) - n_in_cls
            in_cls_score *= n_in_cls / (n_in_cls + n_out_of_cls)
            out_of_cls_score *= n_out_of_cls / (n_in_cls + n_out_of_cls)

        return in_cls_score, out_of_cls_score


def compose_score(
    scorer: Scorer,
    transformation: Callable[[float], float],
    range: Tuple[float, float],
    name: str,
) -> Scorer:
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    .. code-block:: python
       :caption: Example usage

       sigmoid = lambda x: 1/(1+np.exp(-x))
       compose_score(Scorer("r2"), sigmoid, range=(0,1), name="squashed r2")

    :param scorer: The object to be composed.
    :param transformation: A scalar transformation
    :param range: The range of the transformation. This will be used e.g. by
        :class:`~pydvl.utils.utility.Utility` for the range of the composed.
    :param name: A string representation for the composition, for `str()`.
    :return: The composite :class:`Scorer`.
    """

    class CompositeScorer(Scorer):
        def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
            score = self._scorer(model=model, X=X, y=y)
            return transformation(score)

    return CompositeScorer(scorer, range=range, name=name)


def _sigmoid(x: float) -> float:
    result: float = expit(x).item()
    return result


squashed_r2 = compose_score(Scorer("r2"), _sigmoid, (0, 1), "squashed r2")
""" A scorer that squashes the R² score into the range [0, 1] using a sigmoid."""


squashed_variance = compose_score(
    Scorer("explained_variance"), _sigmoid, (0, 1), "squashed explained variance"
)
""" A scorer that squashes the explained variance score into the range [0, 1] using
    a sigmoid."""
