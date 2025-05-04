from __future__ import annotations

import functools
import operator
import pickle
from itertools import permutations

import cloudpickle
import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.utils.array import try_torch_import
from pydvl.utils.status import Status
from pydvl.valuation.result import LogResultUpdater, ValuationResult, ValueItem
from pydvl.valuation.types import ValueUpdate


@pytest.fixture
def dummy_values(values, names):
    return ValuationResult(
        algorithm="dummy_valuator",
        status=Status.Converged,
        values=np.array(values),
        variances=np.zeros_like(values),
        data_names=names,
        sort=True,
    )


def test_valueitem_comparison():
    v1 = ValueItem(idx=0, name="a", value=1.0, variance=0.0, count=1)
    v2 = ValueItem(idx=0, name="a", value=1.0, variance=0.0, count=1)
    v3 = ValueItem(idx=1, name="b", value=2.0, variance=0.0, count=1)
    v4 = ValueItem(idx=1, name="b", value=3.0, variance=0.0, count=1)

    assert v1 == v2
    assert v1 != v3
    assert v3 < v4
    assert v4 > v3
    assert v3 <= v4
    assert v4 >= v3

    with pytest.raises(TypeError, match="Cannot compare ValueItem with"):
        v1 == 1.0  # noqa


def test_creation():
    values = np.array([1.0, 2.0, 3.0])
    indices = np.array([0, 1, 2])
    v = ValuationResult(values=values, indices=indices)
    assert len(v) == len(values)
    assert np.all(v.values == values)
    assert np.all(v.indices == indices)

    v = ValuationResult(values=values, indices=indices, sort=False)
    assert len(v) == len(values)
    assert np.all(v.values == values[::-1])
    assert np.all(v.indices == indices[::-1])

    with pytest.raises(
        ValueError, match=r"Lengths of values \(3\) and indices \(2\) do not match"
    ):
        v = ValuationResult(values=values, indices=np.array([0, 1]))

    with pytest.raises(
        ValueError, match=r"Lengths of values \(3\) and variances \(2\) do not match"
    ):
        v = ValuationResult(values=values, variances=np.array([0, 1]))

    with pytest.raises(
        ValueError, match=r"Lengths of values \(3\) and data_names \(1\) do not match"
    ):
        v = ValuationResult(values=values, data_names=["d"])

    with pytest.raises(ValueError, match="Data names must be unique"):
        v = ValuationResult(values=values, data_names=["a", "b", "a"])


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_sorting(values, names, ranks_asc, dummy_values, reverse: bool):
    tmp = dummy_values.sort(reverse=reverse, key="value", inplace=True)
    assert tmp is dummy_values

    assert np.all([it.value for it in dummy_values] == sorted(values, reverse=reverse))
    if reverse:
        ranks_asc = list(reversed(ranks_asc))
    assert np.all(dummy_values.indices == ranks_asc)

    tmp = dummy_values.sort(key="index")
    assert tmp is not dummy_values
    assert np.all(tmp.indices == list(range(len(values))))
    assert np.all([it.value for it in tmp] == values)


@pytest.mark.parametrize("sort", [None, False, True])
def test_positions_sorting(sort):
    v = ValuationResult(
        values=np.random.random(10), indices=np.arange(10, 20), sort=sort
    )
    data_indices = np.array([10, 13, 15])
    assert np.all(v.indices[v.positions(data_indices)] == data_indices)


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_dataframe_sorting(values, names, ranks_asc, dummy_values):
    sorted_names = [names[r] for r in ranks_asc]
    try:
        import pandas  # noqa: F401

        df = dummy_values.to_dataframe(use_names=False)
        assert np.all(df.index.values == ranks_asc)

        df = dummy_values.to_dataframe(use_names=True)
        assert np.all(df.index.values == sorted_names)
        assert np.all(df["dummy_valuator"].values == sorted(values))

        dummy_values.sort(reverse=True, inplace=True)
        df = dummy_values.to_dataframe(use_names=True)
        assert np.all(df.index.values == list(reversed(sorted_names)))
        assert np.all(df["dummy_valuator"].values == sorted(values, reverse=True))
    except ImportError:
        pass


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_iter(names, ranks_asc, dummy_values):
    for rank, it in enumerate(dummy_values):
        assert it.idx == ranks_asc[rank]

    for rank, it in enumerate(dummy_values):
        assert it.name == names[ranks_asc[rank]]


@pytest.mark.parametrize(
    "values, names, ranks_asc", [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])]
)
def test_todataframe(ranks_asc, dummy_values):
    df = dummy_values.to_dataframe()
    assert "dummy_valuator" in df.columns
    assert "dummy_valuator_variances" in df.columns
    assert "dummy_valuator_counts" in df.columns
    assert np.all(df.index.values == ranks_asc)

    df = dummy_values.to_dataframe(column="val")
    assert "val" in df.columns
    assert "val_variances" in df.columns
    assert "val_counts" in df.columns
    assert np.all(df.index.values == ranks_asc)

    df = dummy_values.to_dataframe(use_names=True)
    assert np.all(df.index.values == [it.name for it in dummy_values])


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2.0, 3.0, 1.0, 6.0], ["a", "b", "c", "d"], [2, 0, 1, 3])],
)
def test_indexing(ranks_asc, dummy_values):
    if len(ranks_asc) == 0:
        with pytest.raises(IndexError):
            dummy_values[1]  # noqa
        empty_slice = dummy_values[:2]  # noqa
        assert isinstance(empty_slice, ValuationResult)
        assert len(empty_slice) == 0
    else:
        # Test that indexing returns ValuationResult objects
        single_idx_result = dummy_values[0]
        assert isinstance(single_idx_result, ValuationResult)
        assert len(single_idx_result) == 1
        assert single_idx_result.indices[0] == ranks_asc[0]

        # Test list indexing
        list_idx_result = dummy_values[[0]]
        assert isinstance(list_idx_result, ValuationResult)
        assert len(list_idx_result) == 1
        assert list_idx_result.indices[0] == ranks_asc[0]

        # Test slice indexing
        slice_idx_result = dummy_values[:2]
        assert isinstance(slice_idx_result, ValuationResult)
        assert len(slice_idx_result) == 2
        assert np.all(slice_idx_result.indices == ranks_asc[:2])

        # Test multiple indexing
        multi_idx_result = dummy_values[[0, 1]]
        assert isinstance(multi_idx_result, ValuationResult)
        assert len(multi_idx_result) == 2
        assert np.all(multi_idx_result.indices == ranks_asc[:2])

        # Test negative indexing
        neg_slice_result = dummy_values[-2:]
        assert isinstance(neg_slice_result, ValuationResult)
        assert len(neg_slice_result) == 2
        assert np.all(neg_slice_result.indices == ranks_asc[-2:])

        # Test negative list indexing
        neg_list_result = dummy_values[[-2, -1]]
        assert isinstance(neg_list_result, ValuationResult)
        assert len(neg_list_result) == 2
        assert np.all(neg_list_result.indices == ranks_asc[-2:])

        # Verify metadata is copied
        assert single_idx_result.algorithm == dummy_values.algorithm
        assert single_idx_result.status == dummy_values.status


def test_get_idx():
    """Test getting by data index"""
    values = np.array([5.0, 2.0, 3.0])
    indices = np.array([3, 4, 2])
    result = ValuationResult(values=values, indices=indices)
    with pytest.raises(IndexError):
        result.get(5)
    for v, idx in zip(values, indices):
        assert v == result.get(idx).value
    result.sort(inplace=True)
    for v, idx in zip(values, indices):
        assert v == result.get(idx).value


def test_updating():
    """Tests updating of values.
    Variance updates use Bessel's correction, same as np.var(ddof=1) since we are
    working with sample estimates.
    """

    v = ValuationResult(values=np.array([1.0, 2.0]))
    updater = LogResultUpdater(v)
    updater.process(ValueUpdate(0, np.log(1.0), 1))
    np.testing.assert_allclose(v.counts, [2, 1])
    np.testing.assert_allclose(v.values, [1, 2])
    np.testing.assert_allclose(v.variances, [0.0, 0.0])

    saved = updater.process(ValueUpdate(1, np.log(4.0), 1)).copy()
    np.testing.assert_allclose(v.counts, [2, 2])
    np.testing.assert_allclose(v.values, [1, 3])
    np.testing.assert_allclose(v.variances, [0.0, 2.0])

    updater.process(ValueUpdate(1, np.log(3.0), 1))
    np.testing.assert_allclose(v.counts, [2, 3])
    np.testing.assert_allclose(v.values, [1, 3])
    np.testing.assert_allclose(v.variances, [0.0, 1])

    # Test init updater with counts and variances already set
    other_updater = LogResultUpdater(saved)
    other_updater.process(ValueUpdate(1, np.log(3.0), 1))
    np.testing.assert_allclose(saved.values, [1, 3])
    np.testing.assert_allclose(saved.counts, [2, 3])
    np.testing.assert_allclose(saved.variances, [0.0, 1])

    # Test after sorting
    v.sort(reverse=True, key="value", inplace=True)
    np.testing.assert_allclose(v.counts, [3, 2])
    np.testing.assert_allclose(v.values, [3, 1])
    np.testing.assert_allclose(v.variances, [1, 0.0])

    updater.process(ValueUpdate(0, np.log(1.0), 1))
    np.testing.assert_allclose(v.counts, [3, 3])
    np.testing.assert_allclose(v.values, [3, 1])
    np.testing.assert_allclose(v.variances, [1, 0])

    # Test data indexing
    v = ValuationResult(values=np.array([3.0, 0.0]), indices=np.array([3, 4]))
    updater = LogResultUpdater(v)
    updater.process(ValueUpdate(4, np.log(1.0), 1))
    np.testing.assert_allclose(v.counts, [1, 2])
    np.testing.assert_allclose(v.values, [3, 0.5])
    np.testing.assert_allclose(v.variances, [0.0, 0.5])

    with pytest.raises(IndexError, match="not found in ValuationResult"):
        updater.process(ValueUpdate(5, np.log(1.0), 1))


def test_updating_order_invariance():
    updates = [0.8, 0.9, 1.0, 1.1, 1.2]
    values = []
    for permutation in permutations(updates):
        v = ValuationResult.zeros(indices=np.array([0]))
        updater = LogResultUpdater(v)
        for update in permutation:
            updater.process(ValueUpdate(0, update, 1))
        values.append(v)

    v1 = values[0]
    for v2 in values[1:]:
        np.testing.assert_almost_equal(v1.values, v2.values)


@pytest.mark.parametrize(
    "serialize, deserialize",
    [(pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)],
)
@pytest.mark.parametrize("values, names", [([], None), ([2.0, 3.0, 1.0], None)])
def test_serialization(serialize, deserialize, dummy_values):
    serded = deserialize(serialize(dummy_values))
    assert dummy_values == serded  # Serialization OK (if __eq__ ok...)
    if len(dummy_values) > 0:
        # Sorting only has an effect over equality on non-empty results
        dummy_values.sort(reverse=True, inplace=True)
        assert dummy_values != serded  # Order checks


@pytest.mark.parametrize("values, names", [([], []), ([2, 3, 1], ["a", "b", "c"])])
def test_copy_and_equality(values, names, dummy_values):
    assert dummy_values == dummy_values

    c = dummy_values.copy()
    dummy_values.sort(reverse=True, inplace=True)

    if len(c) > 0:  # Sorting only has an effect over equality on non-empty results
        assert c != dummy_values

    c2 = ValuationResult(
        algorithm="dummy",
        status=c.status,
        values=c.values,
        variances=c._variances,
        data_names=c._names,
    )
    assert c != c2

    c2 = ValuationResult(
        algorithm=c._algorithm,
        status=Status.Failed,
        values=c.values,
        variances=c.variances,
        data_names=c.names,
    )
    assert c != c2

    c2 = ValuationResult(
        algorithm=c._algorithm,
        status=c._status,
        values=c._values,
        variances=c._variances,
        data_names=c._names,
    )
    c2.sort(reverse=not c._sort_order, inplace=True)

    assert c == c2

    if len(c) > 0:
        c2 = ValuationResult(
            algorithm=c._algorithm,
            status=c._status,
            values=c._values + 1.0,
            variances=c._variances,
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
    # Making sure that the str dunder method works when using extra values
    repr_string = str(result)
    for k, v in extra_values.items():
        assert k in repr_string


@pytest.mark.parametrize(
    "extra_values", [{"test_value": 1.2}, {"test_value1": 1.2, "test_value2": "test"}]
)
def test_extra_values_preserved_in_indexing(extra_values):
    """Test that extra values are preserved when indexing."""
    kwargs = dict(
        algorithm="test",
        status=Status.Converged,
        values=np.random.rand(10),
        sort=True,
    )
    kwargs.update(extra_values)
    result = ValuationResult(**kwargs)

    # Test single indexing
    single_result = result[0]
    for k, v in extra_values.items():
        assert getattr(single_result, k) == v

    # Test slice indexing
    slice_result = result[:5]
    for k, v in extra_values.items():
        assert getattr(slice_result, k) == v

    # Test list indexing
    list_result = result[[0, 1, 2]]
    for k, v in extra_values.items():
        assert getattr(list_result, k) == v


def test_set_method():
    """Test the set method for setting ValueItems by data index."""
    values = np.array([1.0, 2.0, 3.0])
    indices = np.array([10, 20, 30])
    result = ValuationResult(values=values, indices=indices)

    new_item = ValueItem(
        idx=20,  # Must match an existing index
        name="test_name",
        value=5.0,
        variance=0.5,
        count=3,
    )

    result.set(20, new_item)
    assert result.get(20) == new_item

    mismatched_item = ValueItem(
        idx=50,  # Doesn't match data_idx we're setting
        name="mismatch",
        value=9.0,
        variance=1.0,
        count=1,
    )

    with pytest.raises(ValueError, match="doesn't match the provided data_idx"):
        result.set(20, mismatched_item)

    with pytest.raises(IndexError, match="not found in ValuationResult"):
        result.set(50, mismatched_item)


def test_get_and_setitem():
    r1 = ValuationResult(
        indices=np.array([10, 20, 30, 40]),
        values=np.array([1.0, 2.0, 3.0, 4.0]),
        data_names=np.array(["a", "b", "c", "d"]),
    )

    r2 = ValuationResult(
        indices=np.array([50, 60]),
        values=np.array([5.0, 6.0]),
        data_names=np.array(["e", "f"]),
    )

    r1[:2] = r2
    # test indexing with slices and iterables too
    assert r1[slice(0, 2)] == r2[[0, 1]]

    # Original state at other positions should be unchanged
    assert all(r1.indices[2:] == [30, 40])  # noqa
    assert all(r1.values[2:] == [3.0, 4.0])  # noqa
    assert all(r1.names[2:] == ["c", "d"])  # noqa

    r2.sort(reverse=True, key="value", inplace=True)
    r1[[0, 1]] = r2
    assert all(r1.indices[:2] == [60, 50])  # noqa
    assert all(r1.values[:2] == [6.0, 5.0])  # noqa
    assert all(r1.names[:2] == ["f", "e"])  # noqa

    with pytest.raises(ValueError, match="Operation would result in duplicate indices"):
        r1[2:] = r2

    with pytest.raises(ValueError, match="Operation would result in duplicate names"):
        r1[0] = ValuationResult(
            indices=np.array([80]), values=np.array([8.0]), data_names=np.array(["e"])
        )

    r1.sort(key="index", inplace=True)
    r3 = r1[::-1]
    r1.sort(reverse=True, key="index", inplace=True)
    assert r3 == r1

    # Test negative indexing
    r3[-1] = r1[3]
    r3[-2] = r1[2]
    assert r3 == r1

    assert r3[3] == r1[-1]
    assert r3[2] == r1[-2]

    # Test error when lengths don't match
    with pytest.raises(ValueError, match="Cannot set .* positions"):
        r1[:3] = r2

    # Test error when not ValuationResult
    with pytest.raises(TypeError, match="Value must be a ValuationResult"):
        r1[0] = ValueItem(idx=0, name="test", value=1.0, variance=0.0, count=1)  # type: ignore

    # Test index types
    with pytest.raises(TypeError, match="Indices must be"):
        r1["a"]  # noqa

    with pytest.raises(TypeError, match="Indices must be"):
        r1["a"] = r2[0]  # noqa

    with pytest.raises(IndexError, match=r"Index 5 out of range \(0, 4\)"):
        r1[5]  # noqa

    with pytest.raises(IndexError, match=r"Index 5 out of range \(0, 4\)"):
        r1[5] = r2[0]


@pytest.mark.parametrize("size", [1, 10])
@pytest.mark.parametrize("total", [None, 1.0, -1.0])
def test_from_random_creation(size: int, total: float | None):
    result = ValuationResult.from_random(size=size, total=total)
    assert len(result) == size
    assert result.status == Status.Converged
    assert result.algorithm == "random"
    if total is not None:
        np.testing.assert_allclose(np.sum(result.values), total, atol=1e-5)


def test_from_random_creation_errors():
    with pytest.raises(ValueError):
        ValuationResult.from_random(size=0)


def test_addition_compatibility():
    v1 = ValuationResult.from_random(size=4)
    with pytest.raises(TypeError, match="Cannot combine ValuationResult with"):
        v1 += 1
    v2 = ValuationResult.from_random(size=4, algorithm="blah")  # noqa
    with pytest.raises(ValueError, match="Cannot combine results from"):
        v1 += v2


def test_equality_compatibility():
    v = ValuationResult.empty(algorithm="foo", status=Status.Pending)  # noqa

    with pytest.raises(TypeError, match="Cannot compare"):
        v == 1  # noqa

    assert not v == ValuationResult.from_random(3)
    assert not v == ValuationResult.empty(algorithm="bar")
    assert not v == ValuationResult.empty(algorithm="foo", status=Status.Converged)  # noqa


def test_adding_random():
    """Test adding multiple valuation results together.

    First we generate a matrix of values, then we split it into multiple subsets
    and create a valuation result for each subset. Then we add all the valuation
    results together and check that the resulting means and variances match with
    those of the original matrix.
    """
    n_data, n_values, n_splits = 10, 1000, 12
    values = np.random.rand(n_data, n_values)
    split_indices = np.sort(np.random.randint(1, n_values, size=n_splits - 1))
    splits = np.split(values, split_indices, axis=1)
    vv = [
        ValuationResult(
            algorithm="dummy",
            status=Status.Pending,
            values=np.average(s, axis=1),
            variances=np.var(s, axis=1),
            counts=np.full(s.shape[0], fill_value=s.shape[1]),
        )
        for s in splits
    ]
    result: ValuationResult = functools.reduce(operator.add, vv)

    true_means = values.mean(axis=1)
    true_variances = values.var(axis=1)
    true_stderr = values.std(axis=1) / np.sqrt(n_values)

    np.testing.assert_allclose(true_means[result.indices], result.values, atol=1e-5)
    np.testing.assert_allclose(
        true_variances[result.indices], result.variances, atol=1e-5
    )
    np.testing.assert_allclose(true_stderr[result.indices], result.stderr, atol=1e-5)


@pytest.mark.parametrize(
    "indices_1, names_1, values_1, indices_2, names_2, values_2,"
    "expected_indices, expected_names, expected_values",
    [  # Disjoint indices
        (
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            [3, 4, 5],
            ["d", "e", "f"],
            [3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            ["a", "b", "c", "d", "e", "f"],
            [0, 1, 2, 3, 4, 5],
        ),
        # Overlapping indices (recall that values are running averages)
        (
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            [1, 2, 3],
            ["b", "c", "d"],
            [3, 4, 5],
            [0, 1, 2, 3],
            ["a", "b", "c", "d"],
            [0, 2, 3, 5],
        ),
        # Overlapping indices with different lengths
        (
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            [1, 2],
            ["b", "c"],
            [3, 4],
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 2, 3],
        ),
        # Overlapping indices with different lengths, change order
        (
            [1, 2],
            ["b", "c"],
            [3, 4],
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 2, 3],
        ),
    ],
)
def test_adding_different_indices(
    indices_1,
    names_1,
    values_1,
    indices_2,
    names_2,
    values_2,
    expected_indices,
    expected_names,
    expected_values,
):
    """Test adding valuation results with disjoint indices"""
    v1 = ValuationResult(
        indices=np.array(indices_1), values=np.array(values_1), data_names=names_1
    )
    v2 = ValuationResult(
        indices=np.array(indices_2), values=np.array(values_2), data_names=names_2
    )
    v3 = v1 + v2

    np.testing.assert_allclose(v3.indices, np.array(expected_indices))
    np.testing.assert_allclose(v3.values, np.array(expected_values), atol=1e-5)
    assert np.all(v3.names == expected_names)


@pytest.mark.parametrize(
    "indices, index_t, data_names, name_t",
    [
        ([0, 1, 2], np.int64, ["a", "b", "c"], "<U1"),
        ([4, 1, 7], np.int64, [4, 1, 7], np.int64),
        ([4, 1, 7], np.int64, [4, 1, 7], np.float64),
    ],
)
def test_types(indices, index_t, data_names, name_t):
    """Test that types for indices and names are correctly preserved when adding
    valuation results"""

    v = ValuationResult(
        indices=np.array(indices, dtype=index_t),
        values=np.ones(len(indices), dtype=np.float64),
        data_names=np.array(data_names, dtype=name_t),
    )
    assert v.indices.dtype == index_t
    assert v.names.dtype == name_t

    v2 = ValuationResult(
        indices=np.array(indices),
        values=np.ones(len(indices)),
        variances=np.zeros(len(indices)),
        data_names=data_names,
    )
    v += v2
    assert v.indices.dtype == index_t
    assert v.names.dtype == name_t


@pytest.mark.parametrize("data_names", [["a", "b", "c"]])
def test_names(data_names):
    """Test that data names are preserved after addition of results"""

    n = len(data_names)
    v = ValuationResult.from_random(size=n, data_names=data_names)
    v2 = ValuationResult.from_random(size=n, data_names=data_names)

    v += v2
    assert np.all(v.names == np.array(data_names))


@pytest.mark.parametrize("n_samples", [0, 3])
def test_empty(n_samples: int):
    v = ValuationResult.empty(algorithm="test")
    assert len(v) == 0

    v2 = ValuationResult(values=np.arange(n_samples), algorithm="test")
    assert v2 == v + v2
    assert v2 == v2 + v

    v3 = ValuationResult(values=np.arange(n_samples), algorithm="fail")
    with pytest.raises(ValueError, match="Cannot combine results"):
        v3 += v


@pytest.mark.parametrize("n_samples", [0, 3])
def test_zeros(n_samples: int):
    v = ValuationResult.zeros(algorithm="test", size=n_samples)
    assert len(v) == n_samples

    v2 = ValuationResult(values=np.arange(n_samples), algorithm="test")
    assert v2 == v + v2
    assert v2 == v2 + v

    v3 = ValuationResult(values=np.arange(n_samples), algorithm="fail")
    with pytest.raises(ValueError, match="Cannot combine results"):
        v3 += v


@pytest.mark.parametrize("indices", [None, np.array([0, 1, 2])])
def test_scaling(indices: NDArray | None):
    """Tests scaling and how"""
    n = 10
    v = ValuationResult(values=np.arange(n), variances=np.ones(n))
    v2 = v.copy()
    v2.scale(factor=2, data_indices=indices)
    np.testing.assert_allclose(v.indices[indices], v2.indices[indices])
    np.testing.assert_allclose(v.values[indices] * 2, v2.values[indices])
    np.testing.assert_allclose(v.variances[indices] * 4, v2.variances[indices])
    np.testing.assert_allclose(v.counts[indices], v2.counts[indices])


def test_pickle_roundtrip():
    result = ValuationResult.from_random(
        size=5, algorithm="test", status=Status.Pending
    )
    pickled = pickle.dumps(result)
    unpickled = pickle.loads(pickled)
    assert result == unpickled


def test_upgrade_hook_called():
    class UpgradableValuationResult(ValuationResult):
        __version__ = "2.0"

        @classmethod
        def __upgrade_state__(cls, state):
            state["upgraded"] = True
            state["_class_version"] = cls.__version__
            return state

    instance = UpgradableValuationResult.from_random(
        size=5, algorithm="upgrade_test", status=Status.Pending
    )

    # Monkey-patch __getstate__ to simulate a legacy state with an old version.
    def __getstate__(self):
        state = super(UpgradableValuationResult, self).__getstate__()
        state["_class_version"] = "1.0"
        return state

    instance.__getstate__ = __getstate__.__get__(instance, UpgradableValuationResult)

    # Trigger __setstate__ and the upgrade hook.
    # Need to use cloudpickle because of the local monkey patch function
    pickled = cloudpickle.dumps(instance)
    unpickled = cloudpickle.loads(pickled)

    # Verify that the upgrade hook was applied.
    assert unpickled.__dict__.get("upgraded") is True
    assert (
        unpickled.__dict__.get("_class_version")
        == UpgradableValuationResult.__version__
    )
    assert instance == unpickled


def test_version_mismatch_raises():
    instance = ValuationResult.from_random(
        size=5, algorithm="test", status=Status.Pending
    )

    # Monkey-patch __getstate__ to simulate a bad version.
    def __getstate__(self):
        state = self.__dict__
        state["_class_version"] = "bad_version"
        return state

    instance.__getstate__ = __getstate__.__get__(instance, ValuationResult)

    pickled = pickle.dumps(instance)

    with pytest.raises(ValueError, match="Pickled ValuationResult version mismatch"):
        pickle.loads(pickled)


@pytest.mark.torch
def test_tensor_inputs():
    """Test that tensor inputs to ValuationResult raise a TypeError."""
    torch = try_torch_import(require=True)

    tensor_values = torch.tensor([1.0, 2.0, 3.0])
    np_values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult(values=tensor_values)

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult(values=np_values, variances=tensor_values)

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult(values=np_values, counts=tensor_values.long())

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult(values=np_values, indices=tensor_values.long())

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult(values=np_values, data_names=tensor_values)

    with pytest.raises(TypeError, match="ValuationResult requires numpy arrays"):
        ValuationResult.zeros(indices=tensor_values.long())
