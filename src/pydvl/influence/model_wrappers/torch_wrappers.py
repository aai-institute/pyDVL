import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.nn import Softmax, Tanh

    _TORCH_INSTALLED = True
except ImportError:
    _TORCH_INSTALLED = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "TorchLinearRegression",
    "TorchBinaryLogisticRegression",
    "TorchMLP",
    "TorchModel",
]


class TorchLinearRegression(nn.Module, TorchModelBase):
    """
    A simple linear regression model (with bias) f(x)=Ax+b.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        init: Tuple["NDArray[np.float_]", "NDArray[np.float_]"] = None,
    ):
        """
        :param n_input: input to the model.
        :param n_output: output of the model
        :param init A tuple with two matrices, namely A of shape [K, D] and b of shape [K]. If set to None Xavier
        uniform initialization is used.
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        if init is None:
            r = np.sqrt(6 / (n_input + n_output))
            init_A = np.random.uniform(-r, r, size=[n_output, n_input])
            init_b = np.zeros(n_output)
            init = (init_A, init_b)

        self.A = nn.Parameter(
            torch.tensor(init[0], dtype=torch.float64), requires_grad=True
        )
        self.b = nn.Parameter(
            torch.tensor(init[1], dtype=torch.float64), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate A @ x + b using RAM-optimized calculation layout.
        :param x: Tensor [NxD] representing the features x_i.
        :returns A tensor [NxK] representing the outputs y_i.
        """
        return x @ self.A.T + self.b
