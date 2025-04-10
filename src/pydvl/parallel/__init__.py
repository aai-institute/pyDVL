"""
This module provides a common interface to parallelization backends. The list of
supported backends is [here][pydvl.parallel.backends]. Backends should be
instantiated directly and passed to the respective valuation method.

We use executors that implement the [Executor][concurrent.futures.Executor]
interface to submit tasks in parallel.
The basic high-level pattern is:

```python
from pydvl.parallel import JoblibParallelBackend

parallel_backend = JoblibParallelBackend()
with parallel_backend.executor(max_workers=2) as executor:
    future = executor.submit(lambda x: x + 1, 1)
    result = future.result()
assert result == 2
```

Running a map-style job is also easy:

```python
from pydvl.parallel import JoblibParallelBackend

parallel_backend = JoblibParallelBackend()
with parallel_backend.executor(max_workers=2) as executor:
    results = list(executor.map(lambda x: x + 1, range(5)))
assert results == [1, 2, 3, 4, 5]
```
!!! tip "Passsing large objects"
    When running tasks which accept heavy inputs, it is important
    to first use `put()` on the object and use the returned reference
    as argument to the callable within `submit()`. For example:
    ```python
    u_ref = parallel_backend.put(u)
    ...
    executor.submit(task, utility=u)
    ```
    Note that `task()` does not need to be changed in any way:
    the backend will `get()` the object and pass it to the function
    upon invocation.
There is an alternative map-reduce implementation
[MapReduceJob][pydvl.parallel.map_reduce.MapReduceJob] which internally
uses joblib's higher level API with `Parallel()` which then indirectly also
supports the use of Dask and Ray.
"""

# HACK to avoid circular imports
from ..utils.types import *
from .backend import *
from .backends import *
from .config import *
from .futures import *
from .map_reduce import *

if len(ParallelBackend.BACKENDS) == 0:
    raise ImportError("No parallel backend found. Please install ray or joblib.")
