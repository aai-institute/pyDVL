from typing import Dict, Tuple

import numpy as np
import pytest
import ray
from numpy.typing import NDArray
from ray.cluster_utils import Cluster

from pydvl.utils.config import ParallelConfig


@pytest.fixture(scope="module", params=["joblib", "ray-local", "ray-external"])
def parallel_config(request, num_workers):
    if request.param == "joblib":
        yield ParallelConfig(backend="joblib", n_cpus_local=num_workers)
    elif request.param == "ray-local":
        yield ParallelConfig(backend="ray", n_cpus_local=num_workers)
        ray.shutdown()
    elif request.param == "ray-external":
        # Starts a head-node for the cluster.
        cluster = Cluster(
            initialize_head=True,
            head_node_args={
                "num_cpus": num_workers,
            },
        )
        yield ParallelConfig(backend="ray", address=cluster.address)
        ray.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="function")
def dataset_alt_seq_simple(
    request,
) -> Tuple[NDArray[np.float_], NDArray[np.int_], Dict[str, float]]:
    """
    The label set is represented as 0000011100011111, with adjustable left and right
    margins. The left margin denotes the percentage of zeros at the beginning, while the
    right margin denotes the percentage of ones at the end. Accuracy can be efficiently
    calculated using a closed-form solution.
    """
    n_element, left_margin, right_margin = request.param
    x = np.linspace(0, 1, n_element)
    y = ((left_margin <= x) & (x < 0.5)) | ((1 - right_margin) <= x)
    y = y.astype(int)
    x = np.expand_dims(x, -1)
    return x, y, {"left_margin": left_margin, "right_margin": right_margin}
