from typing import Callable, Union

import numpy as np


def conjugate_gradient(
    A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    b: np.darray,
    max_k=10,
    tol=1e-10,
):
    return b
