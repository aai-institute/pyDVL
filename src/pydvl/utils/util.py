import numpy as np
from numpy.typing import NDArray


def arr_or_writeable_copy(arr: NDArray) -> NDArray:
    """Return a copy of ``arr`` if it's not writeable, otherwise return ``arr``.

    :param arr: Array to copy if it's not writeable.
    :return: Copy of ``arr`` if it's not writeable, otherwise ``arr``.
    """
    if not arr.flags.writeable:
        return np.copy(arr)

    return arr
