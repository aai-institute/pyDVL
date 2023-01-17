import functools
import operator
import pickle
from copy import deepcopy

import cloudpickle
import numpy as np
import pytest

from pydvl.value import ValuationResult
from pydvl.utils.status import Status


@pytest.fixture
def dummy_values(values, names):
    return ValuationResult(
        algorithm="dummy_valuator",
        status=Status.Converged,
        values=np.array(values),
        stderr=np.zeros_like(values),
        data_names=names,
        sort=True,
    )


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_sorting(values, names, ranks_asc, dummy_values):

    assert np.alltrue([it.value for it in dummy_values] == sorted(values))
    assert np.alltrue(dummy_values.indices == ranks_asc)
    assert np.alltrue(
        [it.value for it in reversed(dummy_values)] == sorted(values, reverse=True)
    )

    dummy_values.sort(reverse=True)
    assert np.alltrue([it.value for it in dummy_values] == sorted(values, reverse=True))
    assert np.alltrue(dummy_values.indices == list(reversed(ranks_asc)))
    assert np.alltrue(
        dummy_values.values[dummy_values.indices] == sorted(values, reverse=True)
    )


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_dataframe_sorting(values, names, ranks_asc, dummy_values):
    sorted_names = [names[r] for r in ranks_asc]
    try:
        import pandas

        df = dummy_values.to_dataframe(use_names=False)
        assert np.alltrue(df.index.values == ranks_asc)

        df = dummy_values.to_dataframe(use_names=True)
        assert np.alltrue(df.index.values == sorted_names)
        assert np.alltrue(df["dummy_valuator"].values == sorted(values))

        dummy_values.sort(reverse=True)
        df = dummy_values.to_dataframe(use_names=True)
        assert np.alltrue(df.index.values == list(reversed(sorted_names)))
        assert np.alltrue(df["dummy_valuator"].values == sorted(values, reverse=True))
    except ImportError:
        pass


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_iter(names, ranks_asc, dummy_values):
    for rank, it in enumerate(dummy_values):
        assert it.index == ranks_asc[rank]

    for rank, it in enumerate(dummy_values):
        assert it.name == names[ranks_asc[rank]]


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_todataframe(ranks_asc, dummy_values):
    df = dummy_values.to_dataframe()
    assert "dummy_valuator" in df.columns
    assert "dummy_valuator_stderr" in df.columns
    assert np.alltrue(df.index.values == ranks_asc)

    df = dummy_values.to_dataframe(column="val")
    assert "val" in df.columns
    assert "val_stderr" in df.columns
    assert np.alltrue(df.index.values == ranks_asc)


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2.0, 3.0, 1.0, 6.0], ["a", "b", "c", "d"], [2, 0, 1, 3])],
)
def test_indexing(ranks_asc, dummy_values):
    if len(ranks_asc) == 0:
        with pytest.raises(IndexError):
            dummy_values[1]  # noqa
        dummy_values[:2]  # noqa
    else:
        assert ranks_asc[:] == [it.index for it in dummy_values[:]]
        assert ranks_asc[0] == dummy_values[0].index
        assert [ranks_asc[0]] == [it.index for it in dummy_values[[0]]]
        assert ranks_asc[:2] == [it.index for it in dummy_values[:2]]
        assert ranks_asc[:2] == [it.index for it in dummy_values[[0, 1]]]
        assert ranks_asc[:-2] == [it.index for it in dummy_values[:-2]]
        assert ranks_asc[-2:] == [it.index for it in dummy_values[-2:]]
        assert ranks_asc[-2:] == [it.index for it in dummy_values[[-2, -1]]]


@pytest.mark.parametrize(
    "serialize, deserialize",
    [(pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)],
)
@pytest.mark.parametrize("values, names", [([], None), ([2.0, 3.0, 1.0], None)])
def test_serialization(serialize, deserialize, dummy_values):
    serded = deserialize(serialize(dummy_values))
    assert dummy_values == serded  # Serialization OK (if __eq__ ok...)
    dummy_values.sort(reverse=True)
    assert dummy_values != serded  # Order checks


@pytest.mark.parametrize("values, names", [([], []), ([2, 3, 1], ["a", "b", "c"])])
def test_equality(values, names, dummy_values):
    assert dummy_values == dummy_values

    c = deepcopy(dummy_values)
    dummy_values.sort(reverse=True)
    assert c != dummy_values

    c2 = ValuationResult(
        algorithm="dummy",
        status=c.status,
        values=c.values,
        stderr=c._stderr,
        data_names=c._names,
    )
    assert c != c2

    c2 = ValuationResult(
        algorithm=c._algorithm,
        status=Status.Failed,
        values=c.values,
        stderr=c._stderr,
        data_names=c._names,
    )
    assert c != c2

    c2 = ValuationResult(
        algorithm=c._algorithm,
        status=c.status,
        values=c.values,
        stderr=c._stderr,
        data_names=c._names,
    )
    c2.sort(c._sort_order)

    assert c == c2

    if len(c) > 0:
        c2 = ValuationResult(
            algorithm=c._algorithm,
            status=c.status,
            values=c.values + 1.0,
            stderr=c._stderr,
            data_names=c._names,
        )
        assert c != c2


@pytest.mark.parametrize(
    "extra_values", [{"test_value": 1.2}, {"test_value1": 1.2, "test_value2": "test"}]
)
def test_extra_values(extra_values):
    kwargs = dict(
        algorithm="test",
        status=Status.Converged,
        values=np.random.rand(10),
        sort=True,
        test_value=1.2,
    )
    kwargs.update(extra_values)
    result = ValuationResult(**kwargs)
    for k, v in extra_values.items():
        assert getattr(result, k) == v
    # Making sure that the repr dunder method works when using extra values
    repr_string = repr(result)
    for k, v in extra_values.items():
        assert k in repr_string


@pytest.mark.parametrize("size", [0, 1, 10, 500])
def test_from_random_creation(size):
    result = ValuationResult.from_random(size)
    assert len(result) == size


def test_adding():
    n_samples, n_values, n_subsets = 10, 1000, 12
    values = np.random.rand(n_samples, n_values)
    split_indices = np.sort(np.random.randint(1, n_values, size=n_subsets - 1))
    splits = np.split(values, split_indices, axis=1)
    vv = [
        ValuationResult(
            algorithm="dummy",
            status=Status.Pending,
            values=np.average(s, axis=1),
            stderr=np.sqrt(np.var(s, axis=1) / s.shape[1]),
            counts=s.shape[1] * np.ones(n_samples),
        )
        for s in splits
    ]
    result: ValuationResult = functools.reduce(operator.add, vv)

    true_means = values.mean(axis=1)
    true_stderr = np.sqrt(values.var(axis=1) / n_values)

    assert np.allclose(true_means, result.values)
    assert np.allclose(true_stderr, result.stderr)
