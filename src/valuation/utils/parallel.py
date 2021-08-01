"""
OUTDATED COMMENT, FIXME: All value functions return a dictionary of
{index: value} and some status / historical information of the algorithm.
This module provides utility functions to run these in parallel and multiple
times, then gather the results for later processing / reporting.
"""
import inspect
import os
import queue
import multiprocessing as mp

from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Any, Callable, Collection, Generic, List, Optional, Type, \
    TypeVar

T = TypeVar('T')
R = TypeVar('R')
Identity = lambda x: x


def available_cpus():
    from platform import system
    if system() == 'Windows':
        return os.cpu_count()
    return len(os.sched_getaffinity(0))


class MapReduceJob(Generic[T]):
    """ There are probably 77 libraries to do just this """
    _job_id: int
    _run_id: int

    def __call__(self, data: Collection[T], job_id: int, run_id: int):
        raise NotImplementedError()

    def reduce(self, chunks: List[R]) -> R:
        return chunks

    @staticmethod
    # FIXME: Can't make the type for fun more specific,
    #  see https://github.com/python/mypy/issues/5876
    def from_fun(fun: Callable[..., R],
                 reducer: Callable[[List[R]], R] = Identity,
                 job_id_arg: str = None, run_id_arg: str = None) \
            -> 'MapReduceJob':
        """
        :param fun:
        :param reducer: Not actually a reduce() operation but the reduction
            itself. E.g. to join lists of results:

             reducer = lambda x: functools.reduce(operator.add, x, [])

            Or to average returned values:

             reducer = np.mean

        :param job_id_arg: argument name to pass the job id for display
            purposes (e.g. progress bar position). The value passed will be
            job_id + 1 (to allow for a global bar at position=0). Set to None if
            not supported by the function.
        :param run_id_arg: argument name to pass the run id, for display and
            caching purposes. A run id will be passed to fun, and used to
        :return:
        """

        class NewJob(MapReduceJob):
            def __call__(self, data, *args, **kwargs):
                args = dict()

                if run_id_arg is None and \
                        'run_id' not in inspect.signature(fun).parameters.keys():
                    del kwargs['run_id']
                elif run_id_arg is not None:
                    kwargs[run_id_arg] = kwargs['run_id']
                    del kwargs['run_id']

                if job_id_arg is None and \
                        'job_id' not in inspect.signature(fun).parameters.keys():
                    del kwargs['job_id']
                elif job_id_arg is not None:
                    kwargs[job_id_arg] = kwargs['job_id']
                    del kwargs['job_id']

                return fun(data, *args, **kwargs)

            def reduce(self, chunks: List[T]) -> T:
                return reducer(chunks)

        return NewJob()


def make_nested_backend(backend: str = 'loky'):
    """ Creates a joblib backend allowing nested Parallel() calls (which by
    default would use SequentialBackend)

    See https://github.com/joblib/joblib/issues/947
    """

    from importlib import import_module
    m = import_module("joblib._parallel_backends")
    base_name = backend.capitalize() + 'Backend'
    base_cls = getattr(m, base_name)

    def get_nested_backend(self):
        backend = type(self)()
        backend.nested_level = 0
        return backend, None

    return type("Nested" + base_name, (base_cls,),
                dict(get_nested_backend=get_nested_backend))


def chunkify(fun, data, njobs: int, run_id: int) -> List:
    # Splits a list of values into chunks for each job
    n = len(data)
    chunk_size = 1 + n // njobs
    arg_values = [data[i:min(i + chunk_size, n)]
                  for i in data[0:n:chunk_size]]
    for j, vv in enumerate(arg_values, start=1):
        yield delayed(fun)(vv, job_id=j, run_id=run_id + 1)


def map_reduce(fun: MapReduceJob[T],
               data: Collection[T],
               num_jobs: int,
               num_runs: int = 1,
               backend: str = 'loky') -> List:
    """ Takes an embarrassingly parallel fun and runs ot in num_jobs parallel
    jobs, splitting arg into the same number of chunks, one for each job.

    It repeats the process num_runs times, allocating jobs across runs. E.g.
    if num_jobs = 90 and num_runs=10, each whole execution of fun uses 9 jobs,
    with the data split evenly among them. If num_jobs=2 and num_runs=10, two
    cores are used, five times in succession, and each job receives all data.

    Results are aggregated per run using fun.reduce(), but not across runs.

    :param fun: Create with `MapReduceJob.from_fun(fun, reducer, ...)`
    :param data: values to split across jobs
    :param num_jobs: number of parallel jobs to run. Does not accept -1
    :param num_runs: number of times to run fun on the whole data.

     :param backend: 'loky', 'threading', 'multiprocessing', etc.
    """
    if num_jobs == num_runs == 1:
        job_result = fun(data, job_id=1, run_id=1)
        return [fun.reduce([job_result])]

    if num_jobs <= num_runs:
        ret = Parallel(n_jobs=num_jobs)(delayed(fun)(data, job_id=1, run_id=r+1)
                                         for r in range(num_runs))
        # HACK for consistency with fun.reduce()'s expected input format
        ret = [[r] for r in ret]
    else:
        num_jobs_sub = num_jobs // num_runs
        remainder = num_jobs % num_runs
        runs = []
        for run in range(num_runs - remainder):
            if num_jobs_sub > 1:
                job = lambda run_id=run: \
                        Parallel(n_jobs=num_jobs_sub) \
                                (chunkify(fun, data, num_jobs_sub, run_id))
            else:
                job = lambda run_id=run: [fun(data, job_id=1, run_id=run_id)]
            runs.append(job)

        # Repeat for the remainder of num_jobs/num_runs with one more chunk per
        # job in order to use all cores up to num_jobs
        if remainder > 0:
            num_jobs_sub += 1
            for run in range(num_runs - remainder, num_runs):
                runs.append(lambda run_id=run:
                                Parallel(n_jobs=num_jobs_sub)
                                    (chunkify(fun, data, num_jobs_sub, run_id)))

        backend = make_nested_backend(backend)()
        ret = Parallel(n_jobs=num_runs, backend=backend) \
            (delayed(r)() for r in runs)

    for i, r in enumerate(ret):
        ret[i] = fun.reduce(r)

    return ret


class InterruptibleWorker(mp.Process):
    """ A simple consumer worker using two queues.

    To use, subclass and implement `_run(self, task) -> result`. See e.g.
    `ShapleyWorker`, then instantiate using `Coordinator` and the methods
     therein.

     TODO: use shared memory to avoid copying data
     FIXME: I don't need both the abort flag and the None task
     """

    def __init__(self,
                 worker_id: int,
                 tasks: mp.Queue,
                 results: mp.Queue,
                 abort: mp.Value):
        """
        :param worker_id: mostly for display purposes
        :param tasks: queue of incoming tasks for the Worker. A task of `None`
                      signals that there is no more processing to do and the
                      worker should exit after finishing its current task.
        :param results: queue of outgoing results.
        :param abort: shared flag to signal that the worker must stop in the
                      next iteration of the inner loop
        """
        # Mark as daemon, so we are killed when the parent exits (e.g. Ctrl+C)
        super().__init__(daemon=True)

        self.id = worker_id
        self.tasks = tasks
        self.results = results
        self._abort = abort

    def run(self):
        task = self.tasks.get(timeout=2.0)  # Wait a bit during start-up (yikes)
        while True:
            if task is None:  # Indicates we are done
                self.tasks.put(None)
                return

            result = self._run(task)

            # FIXME: I already have the abort flag. Do I need a stop task as
            #  well?
            # Check whether the None flag was sent during _run().
            # This throws away our last results, but avoids a deadlock (can't
            # join the process if a queue has items)
            try:
                task = self.tasks.get_nowait()
                if task is not None:
                    self.results.put(result)
                else:
                    self.tasks.put(None)
            except queue.Empty:
                return

    def aborted(self):
        return self._abort.value is True

    def _run(self, task: Any) -> Any:
        raise NotImplementedError("Please reimplement")


class Coordinator:
    """ Meh... """
    def __init__(self, processor: Callable[[Any], None]):
        self.tasks_q = mp.Queue()
        self.results_q = mp.Queue()
        self.abort_flag = mp.Value('b', False)
        self.workers = []
        self.process_result = processor

    def instantiate(self, n: int, cls: Type[InterruptibleWorker], **kwargs):
        if not self.workers:
            self.workers = [cls(worker_id=i+1,
                                tasks=self.tasks_q,
                                results=self.results_q,
                                abort=self.abort_flag,
                                **kwargs)
                            for i in range(n)]
        else:
            raise ValueError("Workers already instantiated")

    def put(self, task: Any):
        self.tasks_q.put(task)

    def get_and_process(self, timeout: float = None):
        """"""
        try:
            result = self.results_q.get(timeout=timeout)
            self.process_result(result)
        except queue.Empty as e:
            # TODO: do something here?
            raise e

    def clear_tasks(self):
        # Clear the queue of pending tasks
        try:
            while True:
                self.tasks_q.get_nowait()
        except queue.Empty:
            pass

    def clear_results(self, pbar: Optional[tqdm] = None):
        if pbar:
            pbar.set_description_str("Gathering pending results")
            pbar.total = len(self.workers)
            pbar.reset()
        try:
            while True:
                self.get_and_process(timeout=1.0)
                if pbar:
                    pbar.update()
        except queue.Empty:
            pass

    def start(self):
        for w in self.workers:
            w.start()

    def end(self, pbar: Optional[tqdm] = None):
        self.clear_tasks()
        # Any workers still running won't post their results after the
        # None task has been placed...
        self.tasks_q.put(None)
        self.abort_flag.value = True

        # ... But maybe someone put() some result while we were doing the above
        self.clear_results(pbar)

        # results_q should be empty now
        assert self.results_q.empty(), \
            f"WTF? {self.results_q.qsize()} pending results"
        # HACK: the peeking in workers' run() might empty the queue temporarily
        #  between the peeking and the restoring temporarily, so we allow for
        #  some timeout.
        assert self.tasks_q.get(timeout=0.1) is None, \
            f"WTF? {self.tasks_q.qsize()} pending tasks"

        for w in self.workers:
            w.join()
            w.close()
