"""
This module contains the implementation of the
[ClasswiseScorer][pydvl.valuation.scorers.classwise.ClasswiseScorer] class for
Class-wise Shapley values.

TODO: finish doc
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import SupervisedModel
from pydvl.valuation.scorers.scorer import Scorer, ScorerCallable


class ClasswiseScorer(Scorer):
    r"""A Scorer designed for evaluation in classification problems. Its value
    is computed from an in-class and an out-of-class "inner score" (Schoch et
    al., 2022) <sup><a href="#schoch_csshapley_2022">1</a></sup>. Let $S$ be the
    training set and $D$ be the valuation set. For each label $c$, $D$ is
    factorized into two disjoint sets: $D_c$ for in-class instances and $D_{-c}$
    for out-of-class instances. The score combines an in-class metric of
    performance, adjusted by a discounted out-of-class metric. These inner
    scores must be provided upon construction or default to accuracy. They are
    combined into:

    $$
    u(S_{y_i}) = f(a_S(D_{y_i}))\ g(a_S(D_{-y_i})),
    $$

    where $f$ and $g$ are continuous, monotonic functions. For a detailed
    explanation, refer to section four of (Schoch et al., 2022)<sup><a
    href="#schoch_csshapley_2022"> 1</a></sup>.

    !!! warning Multi-class support
        Metrics must support multiple class labels if you intend to apply them
        to a multi-class problem. For instance, the metric 'accuracy' supports
        multiple classes, but the metric `f1` does not. For a two-class
        classification problem, using `f1_weighted` is essentially equivalent to
        using `accuracy`.

    Args:
        scoring: Name of the scoring function or a callable that can be passed
            to [Scorer][pydvl.utils.score.Scorer].
        default: Score to use when a model fails to provide a number, e.g. when
            too little was used to train it, or errors arise.
        range: Numerical range of the score function. Some Monte Carlo methods
            can use this to estimate the number of samples required for a
            certain quality of approximation. If not provided, it can be read
            from the `scoring` object if it provides it, for instance if it was
            constructed with
            [compose_score][pydvl.utils.score.compose_score].
        in_class_discount_fn: Continuous, monotonic increasing function used to
            discount the in-class score.
        out_of_class_discount_fn: Continuous, monotonic increasing function used
            to discount the out-of-class score.
        initial_label: Set initial label (for the first iteration)
        name: Name of the scorer. If not provided, the name of the inner scoring
            function will be prefixed by `classwise `.

    !!! tip "New in version 0.7.1"
    """

    def __init__(
        self,
        scoring: Union[str, ScorerCallable] = "accuracy",
        default: float = 0.0,
        range: Tuple[float, float] = (0, 1),
        in_class_discount_fn: Callable[[float], float] = lambda x: x,
        out_of_class_discount_fn: Callable[[float], float] = np.exp,
        initial_label: Optional[int] = None,
        name: Optional[str] = None,
    ):
        disc_score_in_class = in_class_discount_fn(range[1])
        disc_score_out_of_class = out_of_class_discount_fn(range[1])
        transformed_range = (0, disc_score_in_class * disc_score_out_of_class)
        super().__init__(
            scoring=scoring,
            range=transformed_range,
            default=default,
            name=name or f"classwise {str(scoring)}",
        )
        self._in_class_discount_fn = in_class_discount_fn
        self._out_of_class_discount_fn = out_of_class_discount_fn
        self.label = initial_label

    def __str__(self) -> str:
        return self._name

    def __call__(
        self: "ClasswiseScorer",
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
    ) -> float:
        (
            in_class_score,
            out_of_class_score,
        ) = self.estimate_in_class_and_out_of_class_score(model, x_test, y_test)
        disc_score_in_class = self._in_class_discount_fn(in_class_score)
        disc_score_out_of_class = self._out_of_class_discount_fn(out_of_class_score)
        return disc_score_in_class * disc_score_out_of_class

    def estimate_in_class_and_out_of_class_score(
        self,
        model: SupervisedModel,
        x_test: NDArray[np.float_],
        y_test: NDArray[np.int_],
        rescale_scores: bool = True,
    ) -> Tuple[float, float]:
        r"""
        Computes in-class and out-of-class scores using the provided inner
        scoring function. The result is

        $$
        a_S(D=\{(x_1, y_1), \dots, (x_K, y_K)\}) = \frac{1}{N} \sum_k s(y(x_k), y_k).
        $$

        In this context, for label $c$ calculations are executed twice: once for $D_c$
        and once for $D_{-c}$ to determine the in-class and out-of-class scores,
        respectively. By default, the raw scores are multiplied by $\frac{|D_c|}{|D|}$
        and $\frac{|D_{-c}|}{|D|}$, respectively. This is done to ensure that both
        scores are of the same order of magnitude. This normalization is particularly
        useful when the inner score function $a_S$ is calculated by an estimator of the
        form $\frac{1}{N} \sum_i x_i$, e.g. the accuracy.

        Args:
            model: Model used for computing the score on the validation set.
            x_test: Array containing the features of the classification problem.
            y_test: Array containing the labels of the classification problem.
            rescale_scores: If set to True, the scores will be denormalized. This is
                particularly useful when the inner score function $a_S$ is calculated by
                an estimator of the form $\frac{1}{N} \sum_i x_i$.

        Returns:
            Tuple containing the in-class and out-of-class scores.
        """
        scorer = self._scorer
        label_set_match = y_test == self.label
        label_set = np.where(label_set_match)[0]
        num_classes = len(np.unique(y_test))

        if len(label_set) == 0:
            return 0, 1 / (num_classes - 1)

        complement_label_set = np.where(~label_set_match)[0]
        in_class_score = scorer(model, x_test[label_set], y_test[label_set])
        out_of_class_score = scorer(
            model, x_test[complement_label_set], y_test[complement_label_set]
        )

        if rescale_scores:
            n_in_class = np.count_nonzero(y_test == self.label)
            n_out_of_class = len(y_test) - n_in_class
            in_class_score *= n_in_class / (n_in_class + n_out_of_class)
            out_of_class_score *= n_out_of_class / (n_in_class + n_out_of_class)

        return in_class_score, out_of_class_score
