import pytest

from pydvl.parallel.config import ParallelConfig

from ..conftest import num_workers


@pytest.fixture(scope="module", params=["joblib", "ray-local", "ray-external"])
def parallel_config(request):
    if request.param == "joblib":
        yield ParallelConfig(backend="joblib", n_cpus_local=num_workers())
    elif request.param == "ray-local":
        ray = pytest.importorskip("ray", reason="Ray not installed.")
        ray.init(num_cpus=num_workers())
        yield ParallelConfig(backend="ray")
        ray.shutdown()
    elif request.param == "ray-external":
        ray = pytest.importorskip("ray", reason="Ray not installed.")

        from ray.cluster_utils import Cluster

        # Starts a head-node for the cluster.
        cluster = Cluster(
            initialize_head=True, head_node_args={"num_cpus": num_workers()}
        )
        ray.init(cluster.address)
        yield ParallelConfig(backend="ray", address=cluster.address)
        ray.shutdown()
        cluster.shutdown()
