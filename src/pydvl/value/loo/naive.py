from deprecate import deprecated

from pydvl.utils import Utility
from pydvl.value.result import ValuationResult

from .loo import loo

__all__ = ["naive_loo"]


@deprecated(
    target=loo, deprecated_in="0.7.0", remove_in="0.8.0", args_extra=dict(n_jobs=1)
)
def naive_loo(u: Utility, *, progress: bool = True, **kwargs) -> ValuationResult:
    """Deprecated. Use :func:`~pydvl.value.loo.loo` instead."""
    pass
