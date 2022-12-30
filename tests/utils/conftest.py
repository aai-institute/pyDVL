import pytest

from pydvl.utils.config import ParallelConfig


@pytest.fixture(scope="session", params=["sequential", "ray"])
def parallel_config(request):
    return ParallelConfig(backend=request.param)
