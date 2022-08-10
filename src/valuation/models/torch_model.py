"""
Contains all models used in test and demonstration. Note that they could be written as one module, but for clarity all
 three are defined explicitly.
"""

__all__ = [
    "LinearRegressionTorchModel",
    "BinaryLogisticRegressionTorchModel",
    "NeuralNetworkTorchModel",
]

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Softmax, Tanh


class LinearRegressionTorchModel(nn.Module):
    """
    A simple linear regression model (with bias) f(x)=Ax+b.
    """

    def __init__(self, dim=Tuple[int, int], init: Tuple[np.ndarray, np.ndarray] = None):
        """
        :param dim: A tuple (K, D) representing the size of the model.
        :param init A tuple with two matrices, namely A of shape [K, D] and b of shape [K]. If set to None Xavier
        uniform initialization is used.
        """
        super().__init__()
        n_output, n_input = dim
        self.n_input = n_input
        self.n_output = n_output
        if init is None:
            r = np.sqrt(6 / (n_input + n_output))
            init_A = np.random.uniform(-r, r, size=[n_output, n_input])
            init_b = np.zeros(n_output)
            init = (init_A, init_b)

        self.A = nn.Parameter(
            torch.tensor(init[0], dtype=torch.float32), requires_grad=True
        )
        self.b = nn.Parameter(
            torch.tensor(init[1], dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Calculate A @ x + b using RAM-optimized calculation layout.
        :param x: Tensor [NxD] representing the features x_i.
        :returns A tensor [NxK] representing the outputs y_i.
        """
        return x @ self.A.T + self.b


class BinaryLogisticRegressionTorchModel(nn.Module):
    """
    A simple binary logistic regression model p(y)=sigmoid(dot(a, x) + b).
    """

    def __init__(self, n_input: int, init: Tuple[np.ndarray, np.ndarray] = None):
        """
        :param n_input: Number of feature inputs to the BinaryLogisticRegressionModel.
        :param init A tuple representing the initialization for the weight matrix A and the bias b. If set to None
        sample the values uniformly using the Xavier rule.
        """
        super().__init__()
        self.n_input = n_input
        if init is None:
            init_A = np.random.normal(0, 0.02, size=(1, n_input))
            init_b = np.random.normal(0, 0.02, size=(1))
            init = (init_A, init_b)

        self.A = nn.Parameter(
            torch.tensor(init[0], dtype=torch.float32), requires_grad=True
        )
        self.b = nn.Parameter(
            torch.tensor(init[1], dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Calculate sigmoid(dot(a, x) + b) using RAM-optimized calculation layout.
        :param x: Tensor [NxD] representing the features x_i.
        :returns: A tensor [N] representing the probabilities for p(y_i).
        """
        return torch.sigmoid(x @ self.A.T + self.b)


class NeuralNetworkTorchModel(nn.Module):
    """
    A simple fully-connected neural network f(x) model defined by y = v_K, v_i = o(A v_(i-1) + b), v_1 = x. It contains
    K layers and K - 2 hidden layers. It holds that K >= 2, because every network contains a input and output.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_neurons_per_layer: List[int],
        output_probabilities: bool = True,
        init: List[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        :param n_input: Number of feature inputs to the NeuralNetworkTorchModel.
        :param n_output: Number of outputs to the NeuralNetworkTorchModel or the number of classes.
        :param n_neurons_per_layer: Each integer represents the size of a hidden layer. Overall this list has K - 2
        :param output_probabilities: True, if the model should output probabilities. In the case of n_output 2 the
        number of outputs reduce to 1.
        :param init: A list of tuple of np.ndarray representing the internal weights.
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = 1 if output_probabilities and n_output == 2 else n_output

        self.n_hidden_layers = n_neurons_per_layer
        self.output_probabilities = output_probabilities

        all_dimensions = [self.n_input] + self.n_hidden_layers + [self.n_output]
        layers = []
        num_layers = len(all_dimensions) - 1
        for num_layer, (in_features, out_features) in enumerate(
            zip(all_dimensions[:-1], all_dimensions[1:])
        ):
            linear_layer = nn.Linear(
                in_features, out_features, bias=num_layer < len(all_dimensions) - 2
            )

            if init is None:
                torch.nn.init.xavier_uniform_(linear_layer.weight)
                if num_layer < len(all_dimensions) - 2:
                    linear_layer.bias.data.fill_(0.01)

            else:
                A, b = init[num_layer]
                linear_layer.weight.data = A
                if num_layer < len(all_dimensions) - 2:
                    linear_layer.bias.data = b

            layers.append(linear_layer)
            if num_layer < num_layers - 1:
                layers.append(Tanh())
            elif self.output_probabilities:
                layers.append(Softmax(dim=-1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform forward-pass through the network.
        :param x: Tensor input of shape [NxD].
        :returns: Tensor output of shape[NxK].
        """
        return self.layers(x)
