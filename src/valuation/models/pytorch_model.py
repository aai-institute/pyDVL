from enum import Enum
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

from valuation.utils import maybe_progress
from valuation.utils.types import TorchObjective

tt = partial(torch.tensor, dtype=torch.float32)


def flatten_gradient(grad):
    return torch.cat([el.view(-1) for el in grad])


class PyTorchOptimizer(Enum):
    ADAM = 1
    ADAM_W = 2


class PyTorchSupervisedModel:
    def __init__(
        self,
        model: nn.Module,
        objective: TorchObjective = None,
        optimizer: PyTorchOptimizer = PyTorchOptimizer.ADAM_W,
        optimizer_kwargs: Dict = None,
        num_epochs: int = 1,
        batch_size: int = 64,
    ):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def grad(self, x: np.ndarray, y: np.ndarray, progress: bool = False) -> np.ndarray:

        x = tt(x)
        y = tt(y)

        grads = [
            flatten_gradient(
                autograd.grad(
                    self.objective(self.model(x[i]), y[i]), self.model.parameters()
                )
            )
            .detach()
            .numpy()
            for i in maybe_progress(range(len(x)), progress)
        ]
        return np.stack(grads, axis=0)

    def hvp(
        self, x: np.ndarray, y: np.ndarray, v: np.ndarray, progress: bool = False
    ) -> np.ndarray:

        x, y, v = tt(x), tt(y), tt(v)
        loss = self.objective(self.model(x), y)
        grad_f = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_f = flatten_gradient(grad_f)
        z = (grad_f * Variable(v)).sum(dim=1)
        all_flattened_grads = [
            flatten_gradient(
                autograd.grad(z[i], self.model.parameters(), retain_graph=True)
            )
            for i in maybe_progress(range(len(z)), progress)
        ]
        hvp = torch.stack([grad.contiguous().view(-1) for grad in all_flattened_grads])
        return hvp.detach().numpy()

    def fit(self, x: np.ndarray, y: np.ndarray):
        from valuation.utils.logging import logger

        x = tt(x)
        y = tt(y)

        optimizer_factory = {
            PyTorchOptimizer.ADAM: Adam,
            PyTorchOptimizer.ADAM_W: AdamW,
        }
        optimizer = optimizer_factory[self.optimizer](
            self.model.parameters(), **self.optimizer_kwargs
        )

        class InternalDataset(Dataset):
            def __len__(self):
                return len(x)

            def __getitem__(self, idx):
                return x[idx], y[idx]

        dataset = InternalDataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for epoch in range(self.num_epochs):
            for train_batch in dataloader:
                batch_x, batch_y = train_batch
                pred_y = self.model(batch_x)
                loss = self.objective(pred_y, batch_y)

                logger.debug(f"Training loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model(tt(x)).detach().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        x, y = tt(x), tt(y)
        return self.objective(self.model(x), y).detach().numpy()
