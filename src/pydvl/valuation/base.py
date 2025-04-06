"""
This module declares the abstract base classes for all valuation methods.
A **valuation method** is any function that computes a value for each data point in a
dataset.

!!! info
    For information on data valuation, read [the introduction][data-valuation-intro].
"""

from __future__ import annotations

import io
import logging
import os.path
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from typing_extensions import Self

from pydvl.utils.exceptions import NotFittedException
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult

__all__ = ["Valuation", "ModelFreeValuation"]

logger = logging.getLogger(__name__)


class Valuation(ABC):
    """Abstract base class for all valuation methods."""

    algorithm_name: str = "Valuation"

    def __init__(self) -> None:
        self.result: ValuationResult | None = None

    @abstractmethod
    def fit(self, data: Dataset) -> Self: ...

    def init_or_check_result(self, data: Dataset) -> ValuationResult:
        """Initialize the valuation result or check that a previously initialized one
        (e.g. with `load()`) matches the data.
        """

        if self.result is None:
            return ValuationResult.zeros(
                algorithm=str(self),
                indices=data.indices,
                data_names=data.names,
            )
        else:
            try:
                assert all(self.result.indices == data.indices)
                assert all(self.result.names == data.names)
            except (AssertionError, ValueError) as e:
                raise ValueError(
                    "Either the indices or the names of the dataset do not match those of "
                    "the valuation result. Please reinitialize the valuation method."
                ) from e
        return self.result

    def load(
        self, file: str | os.PathLike | io.IOBase, ignore_exists: bool = True
    ) -> Self:
        """Load the valuation result from a file or file-like object.

        The file or stream must be in the format used by `save()`. If the file does not
        exist, the method does nothing and returns the current instance.

        !!! warning "Temporary solution"
            This simple persistence method is only a temporary solution. It does not
            save any object state other than the result. In particular, interrupting and
            continuing computation from a stored result will not yield the same result
            as uninterrupted computation.

        Args:
            file: The name or path of the file to load, or a file-like object.
            ignore_exists: If `True`, do not raise an error if the file does not exist.
        Raises:
            FileNotFoundError: If the file does not exist and `ignore_exists` is `False`.
            ValueError: If the algorithm of the valuation result does not match the
                current method.
        """
        from joblib import load

        try:
            self.result = load(file)
        except FileNotFoundError as e:
            msg = f"File '{file}' not found. Cannot load valuation result."
            if ignore_exists:
                logger.debug(msg + " Ignoring.")
                return self
            raise FileNotFoundError(msg) from e

        if self.result.algorithm != str(self):
            logger.warning(
                f"The algorithm of the valuation result {self.result.algorithm} does "
                f"not match the current method {str(self)}. Proceed with caution."
            )
        return self

    def save(self, file: str | os.PathLike | io.IOBase) -> Self:
        """Save the valuation result to a file or file-like object.

        The file or stream must be in the format used by `load()`. If the file already
        exists, it will be overwritten.

        Args:
            file: The name or path of the file to save to, or a file-like object.
        """
        from joblib import dump

        if isinstance(file, Path):
            os.makedirs(file.parent, exist_ok=True)
        dump(self.result, file)
        return self

    def values(self, sort: bool = False) -> ValuationResult:
        """Returns a copy of the valuation result.

        The valuation must have been run with `fit()` before calling this method.

        Args:
            sort: Whether to sort the valuation result by value before returning it.
        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise NotFittedException(type(self))
        assert self.result is not None

        r = self.result.copy()
        if sort:
            r.sort()
        return r

    @property
    def is_fitted(self) -> bool:
        return self.result is not None

    def __str__(self):
        return getattr(self, "algorithm_name", self.__class__.__name__)


class ModelFreeValuation(Valuation, ABC):
    """
    TODO: Just a stub, probably should not inherit from Valuation.
    """

    def __init__(self, references: Iterable[Dataset]):
        super().__init__()
        self.datasets = references
