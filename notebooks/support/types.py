from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Losses(NamedTuple):
    training: NDArray[np.float_]
    validation: NDArray[np.float_]
