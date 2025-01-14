from typing import Protocol

import pandas as pd
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.reporting.scores import compute_removal_score
from pydvl.utils.types import BaseModel, ensure_seed_sequence
from pydvl.valuation import Dataset, ModelUtility
from pydvl.valuation.base import Valuation


class ModelFactory(Protocol):
    def __call__(self, random_state: int) -> BaseModel:
        pass


class UtilityFactory(Protocol):
    def __call__(self, *, test: Dataset, random_state: int) -> ModelUtility:
        pass


class ValuationFactory(Protocol):
    def __call__(self, *, train: Dataset, random_state: int) -> Valuation:
        pass


class DataSplitFactory(Protocol):
    def __call__(self, random_state: int) -> tuple[Dataset, Dataset]:
        pass


def removal_job(
    data_factory: DataSplitFactory,
    valuation_factory: ValuationFactory,
    utility_factory: UtilityFactory,
    removal_percentages: NDArray,
    random_state: int,
) -> tuple[dict, dict]:
    """

    Args:
        data_factory:
        valuation_factory:
        utility_factory:
        removal_percentages:
        random_state:
    Returns:


    """

    train, test = data_factory(random_state=random_state)
    valuation = valuation_factory(train=train, random_state=random_state)
    valuation.fit(train)
    values = valuation.values()

    utility = utility_factory(test=test, random_state=random_state)
    low_scores: dict = compute_removal_score(
        utility,
        values,
        train,
        percentages=removal_percentages,
        remove_best=False,
    )
    low_scores["method_name"] = valuation.__class__.__name__

    high_scores: dict = compute_removal_score(
        utility,
        values,
        train,
        percentages=removal_percentages,
        remove_best=True,
    )

    high_scores["method_name"] = valuation.__class__.__name__

    return low_scores, high_scores


def run_removal_experiment(
    data_factory: DataSplitFactory,
    valuation_factories: list[ValuationFactory],
    utility_factory: UtilityFactory,
    removal_percentages: NDArray,
    n_runs: int = 20,
    n_jobs: int = 32,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the sample removal experiment.

    Given the factories, the removal percentages, and the number of runs, this function
    does the following in each run:

    1. Sample a random state
    2. For each valuation method, compute the values and iteratively compute the scores
       after retraining on subsets of the data. This is parallelized. Each job requires
       3 factories:

       - A factory that returns a train-test split of the data given a random state
       - A factory returning a valuation method. The training set is passed to the
         factory, in case the valuation needs to train something. E.g. for Data-OOB
         we need the bagging model to be fitted before the valuation is computed.
       - A factory that returns a utility that evaluates some model on a given test set.
         This is used for the performance evaluation. The model need not be the same
         as the one used for the valuation.
    3. It returns the scores in two DataFrames, one for the high value removals and one
       for the low value removals.

    Args:
        data_factory: A callable that returns a tuple of Datasets (train, test) given
            a random state
        valuation_factories: A list of callables that return Valuation objects given
            a model, train data, and random state. The training data is typically not
            needed for construction, but bagging models may require it
        utility_factory: A callable that returns a ModelUtility given test
            data, potentially with a scoring function
        removal_percentages: The percentage of data to remove from the training set.
            This should be a list of floats between 0 and 1.
        n_runs: The number of repetitions of the experiment
        n_jobs: The number of parallel jobs to use
        random_state: The initial random state
    Returns:
        A tuple of DataFrames with the scores for the low and high value removals
    """
    all_high_scores = []
    all_low_scores = []

    with parallel_config(n_jobs=n_jobs):
        seed_seq = ensure_seed_sequence(random_state).generate_state(n_runs)
        job = delayed(removal_job)

        with Parallel(return_as="generator_unordered") as parallel:
            delayed_evals = parallel(
                job(
                    data_factory=data_factory,
                    valuation_factory=valuation_factory,
                    utility_factory=utility_factory,
                    removal_percentages=removal_percentages,
                    random_state=seed_seq[i],
                )
                for valuation_factory in valuation_factories
                for i in range(n_runs)
            )
            for result in tqdm(
                delayed_evals, unit="%", total=len(valuation_factories) * n_runs
            ):
                low_scores, high_scores = result
                all_low_scores.append(low_scores)
                all_high_scores.append(high_scores)

    low_scores_df = pd.DataFrame(all_low_scores)
    high_scores_df = pd.DataFrame(all_high_scores)

    return low_scores_df, high_scores_df
