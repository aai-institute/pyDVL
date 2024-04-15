from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from joblib.parallel import _get_active_backend

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import IndexSetT

__all__ = ["Valuation", "ModelFreeValuation"]


# FIXME: move these two to a more appropriate place
def ensure_backend_has_generator_return():
    backend = _get_active_backend()[0]
    if not backend.supports_return_generator:
        raise ValueError(
            "The current backend does not support generator return. "
            "Please use a different backend. If you are using ray, follow the "
            "instructions in the documentation to enable generator support."
        )
        # TODO: copy this to the docs:
        #
        #  Instead of calling register_ray(), Do this to enable generator support in
        #  Ray. See:
        #  https://github.com/aai-institute/pyDVL/issues/387#issuecomment-1807994301
        #
        # from ray.util.joblib.ray_backend import RayBackend
        # RayBackend.supports_return_generator = True
        # register_parallel_backend("ray", RayBackend)


def make_parallel_flag(backend=None):
    if backend is None:
        backend = _get_active_backend()[0]
    # raise NotImplementedError

    ## FIXME: temp hack
    class Flag:
        def __init__(self, initial: bool = False):
            self.flag = initial

        def set(self):
            self.flag = True

        def __call__(self):
            return self.flag

    return Flag()


class Valuation(ABC):
    def __init__(self):
        self.result: ValuationResult | None = None

    @abstractmethod
    def fit(self, data: Dataset):
        ...

    def values(self) -> ValuationResult:
        """Returns the valuation result.

        The valuation must have been run with `fit()` before calling this method.

        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise RuntimeError("Valuation is not fitted")
        assert self.result is not None

        return self.result

    @property
    def is_fitted(self) -> bool:
        return self.result is not None


class ModelFreeValuation(Valuation, ABC):
    """
    TODO: Just a stub
    """

    def __init__(self, references: Iterable[Dataset]):
        super().__init__()
        self.datasets = references
