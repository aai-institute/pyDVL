import joblib
import pytest

from pydvl.parallel import JoblibParallelBackend, RayParallelBackend

from ..conftest import num_workers


@pytest.fixture(scope="module", params=["joblib", "ray-local", "ray-external"])
def parallel_backend(request):
    if request.param == "joblib":
        with joblib.parallel_config(backend="loky", n_jobs=num_workers()):
            yield JoblibParallelBackend()
    elif request.param == "ray-local":
        ray = pytest.importorskip("ray", reason="Ray not installed.")
        ray.init(num_cpus=num_workers())
        yield RayParallelBackend()
        ray.shutdown()
    elif request.param == "ray-external":
        ray = pytest.importorskip("ray", reason="Ray not installed.")

        from ray.cluster_utils import Cluster

        # Starts a head-node for the cluster.
        cluster = Cluster(
            initialize_head=True, head_node_args={"num_cpus": num_workers()}
        )
        ray.init(cluster.address)
        yield RayParallelBackend()
        ray.shutdown()
        cluster.shutdown()
