from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from pydvl.utils import CacheBackend, CachedFuncConfig
from pydvl.utils.types import TorchSupervisedModel
from pydvl.valuation import ModelUtility, Scorer
from pydvl.valuation.types import Sample

logger = logging.getLogger(__name__)


class TorchUtility(ModelUtility[Sample, TorchSupervisedModel]):
    """This utility preloads the training data into a PyTorch dataset and moves it to
    the model's device upon the first call.

    This is useful for batched samplers only, since this prefetching will be triggered
    only once at the start of the loop in
    [EvaluationStrategy.process()][pydvl.valuation.evaluation.EvaluationStrategy.process].
    """

    _torch_dataset: TensorDataset | None

    def __init__(
        self,
        model: TorchSupervisedModel,
        scorer: Scorer,
        *,
        catch_errors: bool = False,
        show_warnings: bool = False,
        cache_backend: CacheBackend | None = None,
        cached_func_options: CachedFuncConfig | None = None,
        clone_before_fit: bool = True,
    ):
        super().__init__(
            model,
            scorer,
            catch_errors=catch_errors,
            show_warnings=show_warnings,
            cache_backend=cache_backend,
            cached_func_options=cached_func_options,
            clone_before_fit=clone_before_fit,
        )

    def sample_to_data(self, sample: Sample) -> tuple[Tensor, ...]:
        if getattr(self, "_torch_dataset", None) is None:
            logger.info(
                f"Moving {len(self.training_data)} training data points to {self.model.device}"
            )
            x, y = self.training_data.data()
            x = torch.tensor(x).to(device=self.model.device)
            y = torch.tensor(y).to(device=self.model.device)
            x, y = self.model.reshape_inputs(x, y)
            del self._training_data
            self._torch_dataset = TensorDataset(x, y)

        return self._torch_dataset[sample.subset]
