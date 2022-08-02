"""
Contains all parts of pyTorch based machine learning model. This includes some helper functions, enumerations,
TorchObjective and PyTorchSupervisedModel.
"""

__all__ = [
    "TorchModule",
    "TorchObjective",
    "TorchOptimizer",
]

from enum import Enum
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

from valuation.utils import logger, maybe_progress


def tt(v: Union[np.ndarray, torch.Tensor], dtype: str = "float") -> torch.Tensor:
    """
    Shorthand for torch.Tensor with specific dtype.
    :param v: The tensor to represent as torch.Tensor.
    :param dtype: The dtype to cast the torch.Tensor to.
    :returns: Tensor in torch format.
    """
    if isinstance(v, np.ndarray):
        te = torch.tensor(v)
    else:
        te = v.clone().detach()

    return te.type({"float": torch.float32, "long": torch.long}[dtype])


def flatten_gradient(grad):
    """
    Simple function to flatten a pyTorch gradient for use in subsequent calculation
    """
    return torch.cat([el.view(-1) for el in grad])


class TorchOptimizer(Enum):
    """
    Different optimizer choices for PyTorchSupervisedModel.
    """

    ADAM = 1
    ADAM_W = 2


class TorchObjective:
    """
    Class wraps an objective function along with the data type. It casts
    """

    def __init__(
        self,
        objective: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
        dtype: str = "float",
    ):
        """
        :param objective: A callable which maps two torch.Tensor objects to one and represents the loss function.
        :param dtype: The input to the objective have to be encoded by a dtype, represented as string, e.g. 'long', 'float'.
        """
        self._dtype = dtype
        self._objective = objective

    def __call__(self, y_pred, y_target, **kwargs) -> torch.Tensor:
        return self._objective(y_pred, tt(y_target, self._dtype), **kwargs)


class TorchModule:
    """
    A class which wraps training, gradients and second-derivative matrix vector products of machine learning model into
    a single interface.
    """

    def __init__(
        self,
        model: nn.Module,
        objective: TorchObjective = None,
        optimizer: TorchOptimizer = TorchOptimizer.ADAM_W,
        optimizer_kwargs: Dict[str, Union[str, int, float]] = None,
        num_epochs: int = 1,
        batch_size: int = 64,
    ):
        """
        :param model: A torch.nn.Module representing a (differentiable) function f(x).
        :param objective: Loss function L(f(x), y) maps a prediction and a target to a single value.
        :param optimizer: Select either ADAM or ADAM_W.
        :param optimizer_kwargs: A dictionary with additional optimizer kwargs. This includes parameters 'lr', 'weight_decay' and 'cosine_annealing'.
        :param num_epochs: Number of epochs to repeat training.
        :param batch_size: Batch size to use in training.
        """
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits the model to the supplied data. It represents a simple machine learning loop, iterating over a number of
        epochs, sampling data with a certain batch size, calculating gradients and updating the parameters through a
        loss function. See __init__method for more information.
        :param x: Matrix of shape [NxD] representing the features x_i.
        :param y: Matrix of shape [NxK] representing the prediction targets y_i.
        """
        x = tt(x)
        y = tt(y)

        optimizer_factory = {
            TorchOptimizer.ADAM: Adam,
            TorchOptimizer.ADAM_W: AdamW,
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
            """
            Simple hidden class wrapper for the data loader.
            """

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

                logger.debug(f"Training loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if use_cosine_annealing:
                    scheduler.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Use internal model to deliver prediction in numpy.
        :param x: A np.ndarray [NxD] representing the features x_i.
        :returns: A np.ndarray [NxK] representing the predicted values.
        """
        return self.model(tt(x)).detach().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Use internal model to measure how good is prediction through a loss function.
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :returns: The aggregated value over all samples N.
        """
        x, y = tt(x), tt(y)
        return self.objective(self.model(x), y).detach().numpy()

    def num_params(self) -> int:
        """
        Get number of parameters of model f.
        :returns: Number of parameters as integer.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def grad(self, x: np.ndarray, y: np.ndarray, progress: bool = False) -> np.ndarray:
        """
        Calculates gradient of loss function for tuples (x_i, y_i).
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param progress: True, iff progress shall be printed.
        :returns: A np.ndarray [NxP] representingt the gradients with respect to all parameters of the model.
        """
        x = tt(x)
        y = tt(y)

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
        """
        Calculates matrix vector product of vector v..
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param v: A np.ndarray [NxP] to be mulitplied with the Hessian.
        :param progress: True, iff progress shall be printed.
        :param second_x: True, iff the second dimension should be x of the differnetiation.
        :returns: A np.ndarray [NxP] representingt the gradients with respect to all parameters of the model.
        """

        x, y, v = tt(x), tt(y), tt(v)

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
