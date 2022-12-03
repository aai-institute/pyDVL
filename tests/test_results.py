import numpy as np
import pytest

from pydvl.utils.types import SortOrder
from pydvl.value import ValuationResult, ValuationStatus


def dummy_valuator():
    pass


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])],
)
def test_sorting(values, names, ranks_asc):
    v = ValuationResult(
        dummy_valuator,
        status=ValuationStatus.Converged,
        values=np.array(values),
        stderr=np.zeros_like(values),
        data_names=names,
        sort=SortOrder.Ascending,
    )

    assert np.alltrue([it.value for it in v] == sorted(values))
    assert np.alltrue(v.indices == ranks_asc)

    v.sort(SortOrder.Descending)
    assert np.alltrue([it.value for it in v] == sorted(values, reverse=True))
    assert np.alltrue(v.indices == list(reversed(ranks_asc)))


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])],
)
def test_dataframe_sorting(values, names, ranks_asc):
    v = ValuationResult(
        dummy_valuator,
        status=ValuationStatus.Converged,
        values=np.array(values),
        stderr=np.zeros_like(values),
        data_names=names,
        sort=SortOrder.Ascending,
    )
    sorted_names = [names[r] for r in ranks_asc]
    try:
        import pandas

        df = v.to_dataframe(use_names=False)
        assert np.alltrue(df.index.values == ranks_asc)

        df = v.to_dataframe(use_names=True)
        assert np.alltrue(df.index.values == sorted_names)
        assert np.alltrue(df["dummy_valuator"].values == sorted(values))

        df = v.sort(SortOrder.Descending).to_dataframe(use_names=True)
        assert np.alltrue(df.index.values == list(reversed(sorted_names)))
        assert np.alltrue(df["dummy_valuator"].values == sorted(values, reverse=True))
    except ImportError:
        # FIXME: do we have pandas as strict dependency or not
        pass


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])],
)
def test_iter(values, names, ranks_asc):
    values = ValuationResult(
        dummy_valuator,
        status=ValuationStatus.Converged,
        values=np.array(values),
        stderr=np.zeros_like(values),
        data_names=names,
        sort=SortOrder.Ascending,
    )

    for rank, it in enumerate(values):
        assert it.index == ranks_asc[rank]

    for rank, it in enumerate(values):
        assert it.name == names[ranks_asc[rank]]


@pytest.mark.parametrize(
    "values, names, ranks_asc",
    [([], [], []), ([2, 3, 1], ["a", "b", "c"], [2, 0, 1])],
)
def test_todataframe(values, names, ranks_asc):
    values = ValuationResult(
        dummy_valuator,
        status=ValuationStatus.Converged,
        values=np.array(values),
        stderr=np.zeros_like(values),
        data_names=names,
        sort=SortOrder.Ascending,
    )
    df = values.to_dataframe()
    assert "dummy_valuator" in df.columns
    assert "dummy_valuator_stderr" in df.columns
    assert np.alltrue(df.index.values == ranks_asc)

    df = values.to_dataframe(column="val")
    assert "val" in df.columns
    assert "val_stderr" in df.columns
    assert np.alltrue(df.index.values == ranks_asc)
