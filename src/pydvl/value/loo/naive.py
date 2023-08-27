from deprecate import deprecated

from pydvl.utils import Utility
from pydvl.value.result import ValuationResult

from .loo import compute_loo

__all__ = ["naive_loo"]


@deprecated(
    target=compute_loo,
    deprecated_in="0.7.0",
    remove_in="0.8.0",
    args_extra=dict(n_jobs=1),
)
def naive_loo(u: Utility, *, progress: bool = True, **kwargs) -> ValuationResult:
    """Deprecated. Use [compute_loo][pydvl.value.loo.compute_loo] instead."""
    pass
