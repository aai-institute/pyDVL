from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LRTorchModel(nn.Module):
    def __init__(self, dim=Tuple[int, int], init: Tuple[np.ndarray, np.ndarray] = None):
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
        return x @ self.A.T + self.b
