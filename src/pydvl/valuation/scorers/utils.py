from typing import Callable, Tuple, Type

from scipy.special import expit

from pydvl.utils.types import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.supervised import SupervisedScorer

__all__ = ["compose_score", "squashed_r2", "squashed_variance"]


def compose_score(
    scorer: SupervisedScorer,
    transformation: Callable[[float], float],
    range: Tuple[float, float],
    name: str,
) -> Type[SupervisedScorer]:
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    Example:

    ```python
    sigmoid = lambda x: 1/(1+np.exp(-x))
    compose_score(Scorer("r2"), sigmoid, range=(0,1), name="squashed r2")
    ```

    Args:
        scorer: The object to be composed.
        transformation: A scalar transformation
        range: The range of the transformation. This will be used e.g. by
            [Utility][pydvl.valuation.utility.Utility] for the range of the
            composite scorer.
        name: A string representation for the composition, for `str()`.

    Returns:
        The composite [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer].
    """

    class CompositeSupervisedScorer(SupervisedScorer):
        def __init__(self, test_data: Dataset):
            super().__init__(
                scoring=scorer._scorer,
                test_data=test_data,
                default=transformation(scorer.default),
                range=range,
                name=name,
            )

        def __call__(self, model: SupervisedModel) -> float:
            score = self._scorer(model=model, X=self.test_data.x, y=self.test_data.y)
            return transformation(score)

    return CompositeSupervisedScorer


def _sigmoid(x: float) -> float:
    result: float = expit(x).item()
    return result


# FIXME: yuk, this is awkward...
squashed_r2 = lambda test_data: compose_score(
    SupervisedScorer("r2", test_data, 0), _sigmoid, (0, 1), "squashed r2"
)
""" A scorer that squashes the R² score into the range [0, 1] using a sigmoid."""

squashed_variance = lambda test_data: compose_score(
    SupervisedScorer("explained_variance", test_data, 0),
    _sigmoid,
    (0, 1),
    "squashed explained variance",
)
""" A scorer that squashes the explained variance score into the range [0, 1] using
    a sigmoid."""
