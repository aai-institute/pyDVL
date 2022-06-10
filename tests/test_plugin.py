import pytest


@pytest.mark.tolerate(max_failures=1)
@pytest.mark.parametrize("i", range(1))
def test_marker_with_parametrize(i):
    assert False


@pytest.fixture(scope="function", params=range(2))
def data(request):
    yield


@pytest.mark.xfail
@pytest.mark.tolerate(max_failures=1)
def test_marker_with_data_fixture(data):
    assert False


@pytest.mark.tolerate(max_failures=1)
@pytest.mark.parametrize(
    "i", [range(1), pytest.param(range(2), marks=pytest.mark.xfail)]
)
def test_marker_and_fixture(tolerate, i):
    for _ in i:
        with tolerate:
            assert False


def test_fixture(tolerate):
    for _ in range(1):
        with tolerate(max_failures=1):
            assert False


@pytest.mark.xfail
def test_fixture_call_no_arguments(tolerate):
    for _ in range(1):
        with tolerate():
            assert False


@pytest.mark.xfail()
@pytest.mark.tolerate()
def test_failure():
    pass


@pytest.mark.tolerate(max_failures=0, exceptions_to_ignore=TypeError)
@pytest.mark.parametrize("i", range(5))
def test_marker_ignore_exception(i):
    raise TypeError
