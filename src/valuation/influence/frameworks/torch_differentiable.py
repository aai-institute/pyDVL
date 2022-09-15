"""
Contains all parts of pyTorch based machine learning model.
"""

__all__ = [
    "TorchTwiceDifferentiable",
]

from typing import Any, Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

from valuation.influence.types import TwiceDifferentiable
from valuation.utils import maybe_progress


def flatten_gradient(grad):
    """
    Simple function to flatten a pyTorch gradient for use in subsequent calculation
    """
    return torch.cat([el.view(-1) for el in grad])


class TorchTwiceDifferentiable(TwiceDifferentiable):
    """
    Calculates second-derivative matrix vector products (Mvp) of a pytorch torch.nn.Module
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    ):
        """
        :param model: A torch.nn.Module representing a (differentiable) function f(x).
        :param loss: Loss function L(f(x), y) maps a prediction and a target to a single value.
        """
        self.model = model
        self.loss = loss

    def num_params(self) -> int:
        """
        Get number of parameters of model f.
        :returns: Number of parameters as integer.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def grad(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        progress: bool = False,
    ) -> np.ndarray:
        """
        Calculates gradient of loss function for tuples (x_i, y_i).
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param progress: True, iff progress shall be printed.
        :returns: A np.ndarray [NxP] representing the gradients with respect to all parameters of the model.
        """
        x = torch.as_tensor(x).clone().detach()
        y = torch.as_tensor(y).clone().detach()

        grads = [
            flatten_gradient(
                autograd.grad(
                    self.loss(torch.squeeze(self.model(x[i])), torch.squeeze(y[i])),
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
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        progress: bool = False,
        second_x: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates matrix vector product of vector v.
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param v: A np.ndarray [NxP] to be multiplied with the Hessian.
        :param progress: True, iff progress shall be printed.
        :param second_x: True, iff the second dimension should be x of the differentiation.
        :returns: A np.ndarray [NxP] representing the gradients with respect to all parameters of the model.
        """

        x = torch.as_tensor(x).clone().detach()
        y = torch.as_tensor(y).clone().detach()
        v = torch.as_tensor(v).clone().detach()

        if "num_samples" in kwargs:
            num_samples = kwargs["num_samples"]
            idx = np.random.choice(len(x), num_samples, replace=False)
            x, y = x[idx], y[idx]

        x = nn.Parameter(x, requires_grad=True)
        loss_value = self.loss(torch.squeeze(self.model(x)), torch.squeeze(y))
        grad_f = torch.autograd.grad(
            loss_value, self.model.parameters(), create_graph=True
        )
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
        return hvp.detach().numpy()  # type: ignore
