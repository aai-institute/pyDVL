from typing import Optional

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_dataset(
    test_size: float, val_size: float = 0.0, random_state: Optional[int] = None
):
    """Loads the sklearn handwritten digits dataset. More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset.

    :param test_size: fraction of points used for test dataset
    :param val_size: fraction of points used for training dataset
    :param random_state: fix random seed. If None, no random seed is set.
    :return: A tuple of three elements with the first three being input and
        target values in the form of matrices of shape (N,8,8) the first
        and (N,) the second.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required in order to load the Digits Dataset"
        ) from e

    digits_bunch = load_digits(as_frame=True)
    x, x_test, y, y_test = train_test_split(
        digits_bunch.data.values / 16.0,
        digits_bunch.target.values,
        train_size=1 - test_size,
        random_state=random_state,
    )
    if val_size > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=(1 - val_size) / (1 - test_size), random_state=random_state
        )
    else:
        x_train, y_train = x, y
        x_val, y_val = None, None

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))
