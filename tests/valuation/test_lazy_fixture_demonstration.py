# Here we demonstrate a convenient pytest plugin called lazy fixture.
# It allows to use fixtures within test parametrization, which is especially useful
# if you want to test the method on different input data. Unfortunately, this forces one to use string
# interfaces for fixture but this is a price one might be willing to pay.
import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize(
    "fixture, result",
    [
        (lazy_fixture("fixture_1"), 1),
        (lazy_fixture("fixture_2"), 2),
    ],
)
def test_demonstrateLazyFixture(fixture, result):
    assert fixture == result
