import pytest
import ray
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
