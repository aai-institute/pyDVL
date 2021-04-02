from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Tuple
from joblib import Parallel, delayed
from tqdm import trange


def run_and_gather(fun: Callable[[Iterable], Dict[int, float]],
                   arg: Tuple[str, List],
                   num_jobs: int,
                   num_runs: int) \
        -> List[OrderedDict]:
    """ Runs fun in num_jobs parallel jobs, num_runs times and gathers results
     in a sorted OrderedDict for each run, then returns them in a list.

     All results in one run are merged and sorted by value in ascending order.

    :param fun: A function taking one named argument, with name given in
                `arg[0]`. It accepts a list of values. Each job will receive a
                 chunk of the complete list given in `arg[1]`.
    :param arg: ("fun_arg_name", [values to split across jobs])
    :param num_jobs: number of parallel jobs to run
    :param num_runs: number of times to repeat the whole procedure
    :return: list of length num_runs, containing per-run sorted results
    """
    all_values = []

    # Chunkify the list of values
    n = len(arg[1])
    chunk_size = 1 + int(n / num_jobs)
    values = [list(arg[1][i:min(i + chunk_size, n)])
              for i in arg[1][0:n:chunk_size]]

    jobs = [delayed(fun)(**{arg[0]: vv}) for vv in values]
    for _ in trange(num_runs):
        ret = Parallel(n_jobs=num_jobs)(jobs)
        values = {}
        for _values in ret:
            values.update(_values)

        values = OrderedDict(sorted(values.items(), key=lambda x: x[1]))
        all_values.append(values)
    return all_values
