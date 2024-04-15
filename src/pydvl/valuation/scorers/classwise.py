r"""
This module contains the implementation of the
[ClasswiseScorer][pydvl.valuation.scorers.classwise.ClasswiseScorer] class for
[Class-wise Shapley][pydvl.valuation.methods.classwise_shapley] values.

Its value is computed from an in-class and an out-of-class "inner score" (Schoch et al.,
2022) <sup><a href="#schoch_csshapley_2022">1</a></sup>. Let $S$ be the training set and
$D$ be the valuation set. For each label $c$, $D$ is factorized into two disjoint sets:
$D_c$ for in-class instances and $D_{-c}$ for out-of-class instances. The score combines
an in-class metric of performance, adjusted by a discounted out-of-class metric. These
inner scores must be provided upon construction or default to accuracy. They are
combined into:

$$
u(S_{y_i}) = f(a_S(D_{y_i}))\ g(a_S(D_{-y_i})),
$$

where $f$ and $g$ are continuous, monotonic functions. For a detailed explanation,
refer to section four of (Schoch et al., 2022)<sup><a href="#schoch_csshapley_2022">1</a>
</sup>.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pydvl.utils import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.supervised import (
    SupervisedScorer,
    SupervisedScorerCallable,
)


class ClasswiseSupervisedScorer(SupervisedScorer):
    """A Scorer designed for evaluation in classification problems.

    The final score is the combination of the in-class and out-of-class scores, which
    are e.g. the accuracy of the trained model over the instances of the test set with
    the same, and different, labels, respectively. See the [module's
    documentation][pydvl.valuation.scorers.classwise] for more on this.

    These two scores are computed with an "inner" scoring function, which must be
    provided upon construction.

    !!! warning "Multi-class support"
        The inner score must support multiple class labels if you intend to apply them
        to a multi-class problem. For instance, 'accuracy' supports multiple classes,
        but `f1` does not. For a two-class classification problem, using `f1_weighted`
        is essentially equivalent to using `accuracy`.

    Args:
        scoring: Name of the scoring function or a callable that can be passed
            to [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer].
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
        scoring: str | SupervisedScorerCallable | SupervisedModel,
        test_data: Dataset,
        default: float = 0.0,
        range: tuple[float, float] = (0, 1),
        in_class_discount_fn: Callable[[float], float] = lambda x: x,
        out_of_class_discount_fn: Callable[[float], float] = np.exp,
        initial_label: int | None = None,
        name: str | None = None,
    ):
        disc_score_in_class = in_class_discount_fn(range[1])
        disc_score_out_of_class = out_of_class_discount_fn(range[1])
        transformed_range = (0, disc_score_in_class * disc_score_out_of_class)
        super().__init__(
            scoring=scoring,
            test_data=test_data,
            range=transformed_range,
            default=default,
            name=name or f"classwise {str(scoring)}",
        )
        self._in_class_discount_fn = in_class_discount_fn
        self._out_of_class_discount_fn = out_of_class_discount_fn
        self.label = initial_label
        self.num_classes = len(np.unique(self.test_data.y))

    def __str__(self) -> str:
        return self.name

    def __call__(self, model: SupervisedModel) -> float:
        (in_class_score, out_of_class_score) = self.compute_in_and_out_of_class_scores(
            model
        )
        disc_score_in_class = self._in_class_discount_fn(in_class_score)
        disc_score_out_of_class = self._out_of_class_discount_fn(out_of_class_score)
        return disc_score_in_class * disc_score_out_of_class

    def compute_in_and_out_of_class_scores(
        self, model: SupervisedModel, rescale_scores: bool = True
    ) -> tuple[float, float]:
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
        label_set_match = self.test_data.y == self.label
        label_set = np.where(label_set_match)[0]

        if len(label_set) == 0:
            return 0, 1 / max(1, self.num_classes - 1)

        complement_label_set = np.where(~label_set_match)[0]
        in_class_score = scorer(
            model, self.test_data.y[label_set], self.test_data.y[label_set]
        )
        out_of_class_score = scorer(
            model,
            self.test_data.x[complement_label_set],
            self.test_data.y[complement_label_set],
        )

        if rescale_scores:
            n_in_class = np.count_nonzero(self.test_data.y == self.label)
            n_out_of_class = len(self.test_data.y) - n_in_class
            in_class_score *= n_in_class / (n_in_class + n_out_of_class)
            out_of_class_score *= n_out_of_class / (n_in_class + n_out_of_class)

        return in_class_score, out_of_class_score
