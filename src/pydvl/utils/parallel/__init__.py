"""
This module provides a common interface to parallelization backends. The list of
supported backends is [here][pydvl.utils.parallel.backends].

We use [executors][concurrent.futures.Executor] to submit tasks in parallel. The
basic high-level pattern is

```python
from pydvl.utils.parallel import init_executo
from pydvl.utils.config import ParallelConfig

config = ParallelConfig(backend="ray")
with init_executor(max_workers=1, config=config) as executor:
    future = executor.submit(lambda x: x + 1, 1)
    result = future.result()
assert result == 2
```

Running a map-reduce job is also easy:

```python
from pydvl.utils.parallel import init_executor
from pydvl.utils.config import ParallelConfig

config = ParallelConfig(backend="joblib")
with init_executor() as executor:
    results = list(executor.map(lambda x: x + 1, range(5)))
assert results == [1, 2, 3, 4, 5]
```

There is an alternative map-reduce implementation
[MapReduceJob][pydvl.utils.parallel.map_reduce.MapReduceJob] which internally
uses joblib's higher level API with `Parallel()`
"""
from .backend import *
from .backends import *
from .futures import *
from .map_reduce import *

if len(BaseParallelBackend.BACKENDS) == 0:
    raise ImportError("No parallel backend found. Please install ray or joblib.")
