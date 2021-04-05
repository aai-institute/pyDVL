"""
All value functions return a dictionary of {index: value} and some status /
historical information of the algorithm. This module provides utility functions
to run these in parallel and multiple times, then gather the results for later
processing / reporting.
"""

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Tuple
from joblib import Parallel, delayed
from tqdm import trange


def run_and_gather(fun: Callable[..., Tuple[OrderedDict, List]],
                   num_runs: int,
                   progress: bool = False) \
        -> Tuple[List[OrderedDict], List]:
    """ Runs fun num_runs times and gathers results in a sorted OrderedDict for
    each run, then returns them in a list.

    All results in one run are merged and sorted by value in ascending order.

    :param fun: A callable accepting either no arguments, e.g. a procedure
                wrapped with `valuation.parallel.parallel_wrap()`, which is
                basically `Parallel(delayed(...)(...))(...)`, or exactly one
                integer argument, the run_id, for display purposes.
    :param num_runs: number of times to repeat the whole procedure.
    :param progress: True to display a progress bar at the top of the
                         terminal.
    :return: tuple of 2 lists of length num_runs each, one containing per-run
             results sorted by increasing value, the other historic information
             ("converged") FIXME.
    """

    import inspect
    if 'run_id' not in inspect.signature(fun).parameters.keys():
        _fun = lambda run_id: fun()
    else:
        _fun = fun

    all_values = []
    all_histories = []

    runs = trange(num_runs, position=0) if progress else range(num_runs)
    for i in runs:
        ret = _fun(run_id=i)
        # HACK: Merge results from calls to Parallel
        values = {}
        history = []
        if isinstance(ret, list):
            for vv, hh in ret:
                values.update(vv)
                history.append(hh)
            values = OrderedDict(sorted(values.items(),
                                        key=lambda item: item[1]))
        # Or don't...
        else:
            values, history = ret
        # TODO: checkpoint

        all_values.append(values)
        all_histories.append(history)

    return all_values, all_histories


def parallel_wrap(fun: Callable[[Iterable], Dict[int, float]],
                  arg: Tuple[str, List],
                  num_jobs: int,
                  job_id_arg: str = "job_id") -> Callable:
    """ Wraps an embarrasingly parallelizable fun to run in num_jobs parallel
    jobs, splitting arg into the same number of chunks, one for each job.

    Use later with run_and_gather() to collect results of multiple runs.

    :param fun: A function taking one named argument, with name given in
                `arg[0]`. The argument's value should be a list, given in
                `arg[1]`. Each job will receive a chunk of the complete list.
    :param arg: ("fun_arg_name", [values to split across jobs])
    :param num_jobs: number of parallel jobs to run. Does not accept -1
    :param job_id_arg: argument name to pass the job id for display purposes
                       (e.g. progress bar position). The value passed will be
                       job_id + 1 (to allow for a global bar at position=0). Set
                       to None if not supported by the function.
    """
    # Chunkify the list of values
    n = len(arg[1])
    chunk_size = 1 + int(n / num_jobs)
    values = [list(arg[1][i:min(i + chunk_size, n)])
              for i in arg[1][0:n:chunk_size]]

    if job_id_arg is not None:
        jobs = [delayed(fun)(**{arg[0]: vv, job_id_arg: i + 1})
                for i, vv in enumerate(values)]
    else:
        jobs = [delayed(fun)(**{arg[0]: vv}) for i, vv in enumerate(values)]

    return lambda: Parallel(n_jobs=num_jobs)(jobs)
