from __future__ import annotations

import math
from itertools import chain, takewhile
from typing import Iterable, List, Tuple, cast

import numpy as np
from joblib import Parallel, delayed
from more_itertools import batched
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.types import BatchGenerator, SampleT
from pydvl.valuation.utility.base import UtilityBase

BoolDType = np.bool_


def compute_utility_values_and_sample_masks(
    utility: UtilityBase,
    sampler: IndexSampler,
    n_samples: int,
    progress: bool,
    extra_samples: Iterable[SampleT] | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Calculate utility values and sample masks on samples in parallel.

    Creating the utility evaluations and sample masks is the computational bottleneck
    of several data valuation algorithms, for examples least-core and group-testing.

    Args:
        utility: Utility object with model, data and scoring function.
        sampler: The sampler to use for the valuation.
        n_samples: The number of samples to use from the sampler.
        progress: Whether to show a progress bar.
        extra_samples: Additional samples to evaluate. For example, this can be used
            to calculate the total utility of the dataset in parallel with evaluating
            the utility on the samples. Defaults to None.

    Returns:
        A tuple containing the utility values and the sample masks.

    Raises:
        ValueError: If the utility object does not have training data.

    """
    if utility.training_data is None:
        raise ValueError("Utility object must have training data.")

    indices = utility.training_data.indices
    n_obs = len(indices)

    batch_size = sampler.batch_size
    n_batches = math.ceil(n_samples / batch_size)

    def _create_mask_and_utility_values(
        batch: Iterable[SampleT],
    ) -> tuple[List[NDArray[BoolDType]], List[float]]:
        """Convert sampled indices to boolean masks and calculate utility on each
        sample in batch."""
        masks: List[NDArray[BoolDType]] = []
        u_values: List[float] = []
        for sample in batch:
            m = np.full(n_obs, False)
            m[sample.subset.astype(int)] = True
            masks.append(m)
            u_values.append(utility(sample))

        return masks, u_values

    generator = cast(
        BatchGenerator,
        takewhile(
            lambda _: sampler.n_samples <= n_samples,
            sampler.generate_batches(indices),
        ),
    )

    if extra_samples is not None:
        generator = cast(
            BatchGenerator, chain(generator, batched(extra_samples, batch_size))
        )

    generator_with_progress = cast(
        BatchGenerator,
        tqdm(
            generator,
            disable=not progress,
            total=n_batches - 1,
            position=0,
            desc=f"Preparing {n_samples} constraints",
        ),
    )

    with Parallel(return_as="generator") as parallel:
        results = parallel(
            delayed(_create_mask_and_utility_values)(batch)
            for batch in generator_with_progress
        )

        masks: List[NDArray[BoolDType]] = []
        u_values: List[float] = []
        for m, v in results:
            masks.extend(m)
            u_values.extend(v)

    return np.array(u_values), np.vstack(masks)
