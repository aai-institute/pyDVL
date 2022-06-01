from typing import List
from torch.nn import ReLU

import numpy as np
import torch
import torch.nn as nn


class NNTorchModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_neurons_per_Layer: List[int]):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_layers = n_neurons_per_Layer

        all_dimensions = [self.n_input] + self.n_hidden_layers + [self.n_output]
        layers = []
        for num_layer, (in_features, out_features) in enumerate(
            zip(all_dimensions[:-1], all_dimensions[1:])
        ):
            linear_layer = nn.Linear(
                in_features, out_features, bias=num_layer < len(all_dimensions) - 2
            )
            layers.append(linear_layer)
            if num_layer == len(all_dimensions) - 2:
                layers.append(ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)
