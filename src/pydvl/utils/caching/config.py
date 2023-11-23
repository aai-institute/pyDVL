from dataclasses import dataclass, field
from typing import Collection

__all__ = ["CachedFuncConfig"]


@dataclass
class CachedFuncConfig:
    """Configuration for cached functions and methods, providing
    memoization of function calls.

    Instances of this class are typically used as arguments for the construction
    of a [Utility][pydvl.utils.utility.Utility].

    Args:
        ignore_args: Do not take these keyword arguments into account when
            hashing the wrapped function for usage as key. This allows
            sharing the cache among different jobs for the same experiment run if
            the callable happens to have "nuisance" parameters like `job_id` which
            do not affect the result of the computation.
        time_threshold: Computations taking less time than this many seconds are
            not cached. A value of 0 means that it will always cache results.
        allow_repeated_evaluations: If `True`, repeated calls to a function
            with the same arguments will be allowed and outputs averaged until the
            running standard deviation of the mean stabilizes below
            `rtol_stderr * mean`.
        rtol_stderr: relative tolerance for repeated evaluations. More precisely,
            [memcached()][pydvl.utils.caching.memcached] will stop evaluating the function once the
            standard deviation of the mean is smaller than `rtol_stderr * mean`.
        min_repetitions: minimum number of times that a function evaluation
            on the same arguments is repeated before returning cached values. Useful
            for stochastic functions only. If the model training is very noisy, set
            this number to higher values to reduce variance.
    """

    ignore_args: Collection[str] = field(default_factory=list)
    time_threshold: float = 0
    allow_repeated_evaluations: bool = False
    rtol_stderr: float = 0.1
    min_repetitions: int = 3
