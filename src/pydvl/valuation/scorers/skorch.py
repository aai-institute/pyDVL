from __future__ import annotations

import logging

import torch

from pydvl.valuation.scorers.supervised import SupervisedScorer
from pydvl.valuation.types import SkorchSupervisedModel

__all__ = ["SkorchSupervisedScorer"]

logger = logging.getLogger(__name__)


class SkorchSupervisedScorer(SupervisedScorer[SkorchSupervisedModel, torch.Tensor]):
    """Scorer for Skorch models.

    Because skorch models scorer() requires a numpy array to test against, this
    class moves tensors to cpu before scoring.
    """

    def __call__(self, model: SkorchSupervisedModel) -> float:
        x, y = self.test_data.data()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        return float(self._scorer(model, x, y))
