from __future__ import annotations

import logging

import torch
from torch import Tensor

from pydvl.valuation import TorchSupervisedModel
from pydvl.valuation.scorers.supervised import SupervisedScorer

__all__ = ["TorchModelScorer"]

logger = logging.getLogger(__name__)


class TorchModelScorer(SupervisedScorer[TorchSupervisedModel, torch.Tensor]):
    """Scorer specifically designed for PyTorch models.

    This scorer assumes the input data provided during initialization (`test_data`)
    is already a `torch.Tensor` or can be converted to one. It preloads the test
    data and moves it to the specified model's device upon the first call.

    Important:
        This scorer requires `torch.Tensor` inputs for its `test_data`. If you
        have `numpy.ndarray` data, you are responsible for converting it to a
        `torch.Tensor` *before* initializing this scorer, for example using
        `torch.from_numpy(your_numpy_array)`. Implicit conversion is *not*
        performed.

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

        return float(self._scorer(model, self.x_test, self.y_test))
