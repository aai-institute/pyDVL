from typing import Any, Callable, Tuple

from numpy.typing import NDArray
from scipy.special import expit

from pydvl.utils import SupervisedModel
from pydvl.valuation.scorers.scorer import Scorer

__all__ = ["compose_score", "squashed_r2", "squashed_variance"]


def compose_score(
    scorer: Scorer,
    transformation: Callable[[float], float],
    range: Tuple[float, float],
    name: str,
) -> Scorer:
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
            [Utility][pydvl.valuation.utility.Utility] for the range of the composed.
        name: A string representation for the composition, for `str()`.

    Returns:
        The composite [Scorer][pydvl.utils.score.Scorer].
    """

    class CompositeScorer(Scorer):
        def __call__(
            self, model: SupervisedModel, X: NDArray[Any], y: NDArray[Any]
        ) -> float:
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
