import multiprocessing
import os
import time

import psutil
import pytest

from pydvl.utils.monitor import (
    __monitoring_enabled,
    _memory_monitor_thread,
    end_memory_monitoring,
    start_memory_monitoring,
)


@pytest.fixture(autouse=True)
def cleanup_monitor():
    """
    A fixture to ensure that monitoring is stopped after each test.
    """
    yield
    if __monitoring_enabled.is_set():
        end_memory_monitoring()


def test_double_start(caplog):
    start_memory_monitoring(auto_stop=False)
    # Attempt a second start; should log a warning.
    start_memory_monitoring(auto_stop=False)
    assert "already running" in caplog.text


def test_end_without_start():
    result = end_memory_monitoring()
    assert result == (0, {}), f"Expected (0,{{}}) when not monitoring, got {result}"


def test_thread_cleanup():
    start_memory_monitoring(auto_stop=False)
    time.sleep(0.2)  # Allow some time for the thread to start.

    end_memory_monitoring()
    time.sleep(0.1)  # Wait a bit more to ensure the join has completed.
    thread = _memory_monitor_thread()
    assert thread is not None and not thread.is_alive(), (
        "Monitoring thread should have terminated"
    )


def memory_allocating_child(size_mb) -> int:
    """Child process that allocates approximately 10 MB of memory."""
    data = bytearray(size_mb * 1024 * 1024)
    time.sleep(1)  # Ensure the memory monitor has time to sample.
    return len(data)  # Prevent potential optimization


@pytest.mark.timeout(5)
def test_integration_memory_usage():
    baseline = psutil.Process().memory_info().rss
    start_memory_monitoring(auto_stop=False)

    proc = multiprocessing.Process(target=memory_allocating_child, args=(3,))
    proc.start()
    proc.join()

    peak_mem, mem_usage = end_memory_monitoring()

    mem_increase = peak_mem - baseline
    threshold = 3 * 1024 * 1024
    assert mem_increase >= threshold, (
        f"Expected memory increase of at least 3 MB, but got {mem_increase / 1024 / 1024:.2f} MB"
    )

    total_mem = sum(bytes for _, bytes in mem_usage.items())
    assert total_mem >= peak_mem, (
        "Expected aggregated memory usage to be greater than peak usage"
    )


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="More children reported in CI"
)
@pytest.mark.timeout(5)
def test_integration_multiple_children():
    baseline = psutil.Process().memory_info().rss
    start_memory_monitoring(auto_stop=False)

    processes = [
        multiprocessing.Process(target=memory_allocating_child, args=(1,)),
        multiprocessing.Process(target=memory_allocating_child, args=(3,)),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    peak_mem, mem_usage = end_memory_monitoring()

    mem_increase = peak_mem - baseline
    threshold = 4 * 1024 * 1024
    assert mem_increase >= threshold, (
        f"Expected combined memory increase of at least 4 MB, but got {mem_increase / 1024 / 1024:.2f} MB"
    )
    assert len(mem_usage) == len(processes) + 1, (
        f"Expected memory usage for {len(processes) + 1} processes, but got {len(mem_usage)}"
    )

    total_mem = sum(bytes for _, bytes in mem_usage.items())
    assert total_mem >= peak_mem, (
        "Expected aggregated memory usage to be greater than peak usage"
    )
