import hashlib
from typing import Any

import cloudpickle
from _pytest.cacheprovider import Cache as JsonCache


class CloudPickleCache(JsonCache):
    """This is a cloudpickle version of the pytest cache provider.
    It can be used in fixtures to cache results of heavy computations
    to disk.

    ??? Example
        ```pycon
        @pytest.fixture
        def test_fixture(cache):
            cache_key = "cache_key"
            result = cache.get(cache_key, None)
            if result is None:
                result = heavy_computation()
                cache.set(cache_key, result)
            return result
        ```

        ```pycon
        @pytest.fixture
        def test_fixture_with_args(cache, *args):
            args_hash = cache.hash_arguments(*args)
            cache_key = f"prefix_{args_hash}"
            result = cache.get(cache_key, None)
            if result is None:
                result = heavy_computation()
                cache.set(cache_key, result)
            return result
        ```
    """

    def hash_arguments(*args) -> bytes:
        m = hashlib.sha256()
        for arg in args:
            m.update(str(arg).encode())
        return m.digest()

    def get(self, key: str, default: Any) -> Any:
        """Return the cached value for the given key.

        If no value was yet cached or the value cannot be read, the specified
        default is returned.

        Args:
            key: Must be a ``/`` separated value. Usually the first
                name is the name of your plugin or your application.
            default: The value to return in case of a cache-miss or invalid cache value.
        """
        path = self._getvaluepath(key)
        try:
            with path.open("rb") as f:
                return cloudpickle.load(f)
        except (ValueError, OSError):
            return default

    def set(self, key: str, value: object) -> None:
        """Save value for the given key.

        Args:
            key: Must be a ``/`` separated value. Usually the first
                name is the name of your plugin or your application.
            value: Must be of any combination of basic python types,
                including nested types like lists of dictionaries.
        """
        path = self._getvaluepath(key)
        try:
            if path.parent.is_dir():
                cache_dir_exists_already = True
            else:
                cache_dir_exists_already = self._cachedir.exists()
                path.parent.mkdir(exist_ok=True, parents=True)
        except OSError:
            self.warn("could not create cache path {path}", path=path, _ispytest=True)
            return
        if not cache_dir_exists_already:
            self._ensure_supporting_files()
        try:
            f = path.open("wb")
        except OSError:
            self.warn("cache could not write path {path}", path=path, _ispytest=True)
        else:
            cloudpickle.dump(value, f)
