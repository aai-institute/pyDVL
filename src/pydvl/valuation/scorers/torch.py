from __future__ import annotations

import logging

import torch
from torch import Tensor

from pydvl.valuation import TorchSupervisedModel
from pydvl.valuation.scorers.supervised import SupervisedScorer

__all__ = ["TorchModelScorer"]

logger = logging.getLogger(__name__)


class TorchModelScorer(SupervisedScorer[TorchSupervisedModel, torch.Tensor]):
    """This scorer preloads the test data into a PyTorch dataset and moves it to the
    model's device on first use.

    It is mainly useful for batched samplers only, since this prefetching will be
    triggered only once at the start of the loop in
    [EvaluationStrategy.process()][pydvl.valuation.evaluation.EvaluationStrategy.process].
    """

    x_test: Tensor
    y_test: Tensor

    def __call__(self, model: TorchSupervisedModel) -> float:
        if not hasattr(self, "x_test") or not hasattr(self, "y_test"):
            logger.info(
                f"Moving {len(self.test_data)} test data points to {model.device}"
            )
            x, y = self.test_data.data()
            x = torch.tensor(x).to(device=model.device)
            y = torch.tensor(y).to(device=model.device)
            self.x_test, self.y_test = model.reshape_inputs(x, y)
            del self.test_data

        return self._scorer(model, self.x_test, self.y_test)
