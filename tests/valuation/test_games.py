"""
FIXME: these tests are incomplete
"""

from __future__ import annotations

import pytest

from pydvl.valuation.games import ShoesGame


@pytest.mark.parametrize(
    "n_left, n_right, size, expected",
    [
        (1, 1, 1, 1),
        (1, 1, 2, 0),
        (1, 2, 1, 2),
        (2, 3, 2, 3),
    ],
)
def test_n_subsets_size_k(n_left: int, n_right: int, size: int, expected: int):
    """Test cases are for left shoes, but the problem is symmetric."""
    assert ShoesGame.n_subsets_left(n_left, n_right, size) == expected
    assert ShoesGame.n_subsets_right(n_right, n_left, size) == expected


def n_subsets_left_stratified(n_left: int, n_right: int) -> int:
    acc = 0
    for k in range(1, n_left + n_right + 1):
        acc += ShoesGame.n_subsets_left(n_left, n_right, k)
    return acc


def n_subsets_right_stratified(n_left: int, n_right: int) -> int:
    return n_subsets_left_stratified(n_right, n_left)


CASES = pytest.mark.parametrize(
    "n_left, n_right, expected",
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 4),
        (2, 2, 5),
        (3, 2, 6),
        (4, 2, 7),
        (1, 3, 11),
        (4, 0, 0),
        (7, 1, 1),
        (7, 2, 10),
        (3, 3, 22),
    ],
)


@CASES
def test_n_subsets(n_left: int, n_right: int, expected: int):
    assert ShoesGame.n_subsets_left(n_left, n_right) == expected
    # This should be symmetric
    assert ShoesGame.n_subsets_right(n_right, n_left) == expected


@CASES
def test_n_subsets_stratified(n_left: int, n_right: int, expected: int):
    assert n_subsets_left_stratified(n_left, n_right) == expected
    # This should be symmetric
    assert n_subsets_right_stratified(n_right, n_left) == expected
