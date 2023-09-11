import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

__all__ = ["ParallelConfig"]


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel computation backend.

    Args:
        backend: Type of backend to use. Defaults to 'joblib'
        address: Address of existing remote or local cluster to use.
        n_cpus_local: Number of CPUs to use when creating a local ray cluster.
            This has no effect when using an existing ray cluster.
        logging_level: Logging level for the parallel backend's worker.
        wait_timeout: Timeout in seconds for waiting on futures.
    """

    backend: Literal["joblib", "ray"] = "joblib"
    address: Optional[Union[str, Tuple[str, int]]] = None
    n_cpus_local: Optional[int] = None
    logging_level: int = logging.WARNING
    wait_timeout: float = 1.0

    def __post_init__(self) -> None:
        # FIXME: this is specific to ray
        if self.address is not None and self.n_cpus_local is not None:
            raise ValueError("When `address` is set, `n_cpus_local` should be None.")
