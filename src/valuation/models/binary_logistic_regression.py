from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class BinaryLogisticRegressionTorchModel(nn.Module):
    def __init__(self, n_input: int, init: Tuple[np.ndarray, np.ndarray] = None):
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
        return torch.sigmoid(x @ self.A.T + self.b)
