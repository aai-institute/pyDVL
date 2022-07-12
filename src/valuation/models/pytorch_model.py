from enum import Enum
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

from valuation.utils import logger, maybe_progress
from valuation.utils.types import TorchObjective


def tt(v, dtype=torch.float32):
    return torch.tensor(v, dtype=dtype)


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
        y_dtype=torch.float32,
    ):
        self.model = model
        self.y_dtype = y_dtype
        self.objective = objective
        self.optimizer = optimizer
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def num_params(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def grad(self, x: np.ndarray, y: np.ndarray, progress: bool = False) -> np.ndarray:

        x = tt(x)
        y = torch.tensor(y, dtype=self.y_dtype)

        grads = [
            flatten_gradient(
                autograd.grad(
                    self.objective(
                        torch.squeeze(self.model(x[i])), torch.squeeze(y[i])
                    ),
                    self.model.parameters(),
                )
            )
            .detach()
            .numpy()
            for i in maybe_progress(range(len(x)), progress)
        ]
        return np.stack(grads, axis=0)

    def mvp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        progress: bool = False,
        second_x: bool = False,
        **kwargs,
    ) -> np.ndarray:

        x, y, v = tt(x), torch.tensor(y, dtype=self.y_dtype), tt(v)

        if "num_samples" in kwargs:
            num_samples = kwargs["num_samples"]
            idx = np.random.choice(len(x), num_samples, replace=False)
            x, y = x[idx], y[idx]

        x = nn.Parameter(x, requires_grad=True)
        loss = self.objective(torch.squeeze(self.model(x)), torch.squeeze(y))
        grad_f = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_f = flatten_gradient(grad_f)
        z = (grad_f * Variable(v)).sum(dim=1)
        all_flattened_grads = [
            flatten_gradient(
                autograd.grad(
                    z[i],
                    self.model.parameters() if not second_x else [x],
                    retain_graph=True,
                )
            )
            for i in maybe_progress(range(len(z)), progress)
        ]
        hvp = torch.stack([grad.contiguous().view(-1) for grad in all_flattened_grads])
        return hvp.detach().numpy()

    def fit(self, x: np.ndarray, y: np.ndarray):

        x = tt(x)
        y = torch.tensor(y, dtype=self.y_dtype)

        optimizer_factory = {
            PyTorchOptimizer.ADAM: Adam,
            PyTorchOptimizer.ADAM_W: AdamW,
        }
        optimizer_init_kwargs_keys = {"lr", "weight_decay"}
        reduced_optimizer_kwargs = {
            k: v
            for k, v in self.optimizer_kwargs.items()
            if k in optimizer_init_kwargs_keys
        }
        optimizer = optimizer_factory[self.optimizer](
            self.model.parameters(), **reduced_optimizer_kwargs
        )
        use_cosine_annealing = self.optimizer_kwargs.get("cosine_annealing", False)
        scheduler = None
        if use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs
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
                loss = self.objective(torch.squeeze(pred_y), torch.squeeze(batch_y))

                logger.info(f"Training loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if use_cosine_annealing:
                    scheduler.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model(tt(x)).detach().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        x, y = tt(x), tt(y)
        return self.objective(self.model(x), y).detach().numpy()
