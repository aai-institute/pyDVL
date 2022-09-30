"""
Contains all parts of pyTorch based machine learning model.
"""

__all__ = ["TorchTwiceDifferentiable"]

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

from valuation.influence.types import TwiceDifferentiable
from valuation.utils import maybe_progress

__all__ = [
    "TorchTwiceDifferentiable",
]


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
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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

    def split_grad(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        progress: bool = False,
    ) -> np.ndarray:
        """
        Calculates gradient of model parameters wrt each x[i] and y[i] and then
        returns a array of size [N, P] with N number of points (length of x and y) and P
        number of parameters of the model.
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param progress: True, iff progress shall be printed.
        :returns: A np.ndarray [NxP] representing the gradients with respect to all parameters of the model.
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

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

    def grad(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates gradient of model parameters wrt x and y.
        :param x: A np.ndarray [NxD] representing the features x_i.
        :param y: A np.ndarray [NxK] representing the predicted target values y_i.
        :param progress: True, iff progress shall be printed.
        :returns: A tuple where: \
            - first element is a np.ndarray [P] with the gradients of the model. \
            - second element is the input to the model as a grad parameters. \
                This can be used for further differentiation. 
        """
        x = torch.as_tensor(x).requires_grad_(True)
        y = torch.as_tensor(y)

        loss_value = self.loss(torch.squeeze(self.model(x)), torch.squeeze(y))
        grad_f = torch.autograd.grad(
            loss_value, self.model.parameters(), create_graph=True
        )
        return flatten_gradient(grad_f), x

    def mvp(
        self,
        grad_xy: Union[np.ndarray, torch.Tensor],
        v: Union[np.ndarray, torch.Tensor],
        progress: bool = False,
        backprop_on: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Calculates second order derivative of the model along directions v.
        This second order derivative can be on the model parameters or on another input parameter, 
        selected via the backprop_on argument.

        :param grad_xy: an array [P] holding the gradients of the model parameters wrt input x and labels y, \
            where P is the number of parameters of the model. It is typically obtained through self.grad.
        :param v: A np.ndarray [DxP] which multiplies the Hessian, where D is the number of directions.
        :param progress: True, iff progress shall be printed.
        :param backprop_on: tensor used in the second backpropagation (the first one is along x and y as defined \
            via grad_xy). If None, the model parameters are used.
        :returns: A np.ndarray representing the implicit matrix vector product of the model along the given directions.\
            Output shape is [DxP] if backprop_on is None, otherwise [DxM], with M the number of elements of backprop_on.
        """
        v = torch.as_tensor(v)

        z = (grad_xy * Variable(v)).sum(dim=1)
        all_flattened_grads = [
            flatten_gradient(
                autograd.grad(
                    z[i],
                    self.model.parameters() if backprop_on is None else backprop_on,
                    retain_graph=True,
                )
            )
            for i in maybe_progress(range(len(z)), progress)
        ]
        hvp = torch.stack([grad.contiguous().view(-1) for grad in all_flattened_grads])
        return hvp.detach().numpy()  # type: ignore
