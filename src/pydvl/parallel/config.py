import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

__all__ = ["ParallelConfig"]


# TODO: delete this class once it's made redundant in v0.10.0
@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel computation backend.

    Args:
        backend: Type of backend to use. Defaults to 'joblib'
        address: (DEPRECATED) Address of existing remote or local cluster to use.
        n_cpus_local: (DEPRECATED) Number of CPUs to use when creating a local ray cluster.
            This has no effect when using an existing ray cluster.
        logging_level: (DEPRECATED) Logging level for the parallel backend's worker.
        wait_timeout: (DEPRECATED) Timeout in seconds for waiting on futures.
    """

    backend: Literal["joblib", "ray"] = "joblib"
    address: Optional[Union[str, Tuple[str, int]]] = None
    n_cpus_local: Optional[int] = None
    logging_level: Optional[int] = None
    wait_timeout: float = 1.0

    def __post_init__(self) -> None:
        warnings.warn(
            "The `ParallelConfig` class was deprecated in v0.9.0 and will be removed in v0.10.0",
            FutureWarning,
        )
        if self.address is not None:
            warnings.warn(
                "`address` is deprecated in v0.9.0 and will be removed in v0.10.0",
                FutureWarning,
            )
        if self.n_cpus_local is not None:
            warnings.warn(
                "`n_cpus_local` is deprecated in v0.9.0 and will be removed in v0.10.0",
                FutureWarning,
            )
        if self.logging_level is not None:
            warnings.warn(
                "`logging_level` is deprecated in v0.9.0 and will be removed in v0.10.0",
                FutureWarning,
            )
        if self.wait_timeout != 1.0:
            warnings.warn(
                "`wait_timeout` is deprecated in v0.9.0 and will be removed in v0.10.0",
                FutureWarning,
            )
        # FIXME: this is specific to ray
        if self.address is not None and self.n_cpus_local is not None:
            raise ValueError("When `address` is set, `n_cpus_local` should be None.")
