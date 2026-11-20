from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from gaussfit_rs import fitting as api
from gaussfit_rs import set_workers
from gaussfit_rs._workers import resolve_pool
from gaussfit_rs.tests.test_fit_single_spectrum import COMMON_KW


def run(dtype, **options):
    grid = np.linspace(-300, 300, 60, dtype=np.float32)
    signal = np.rint(3000 * np.exp(-0.5 * (grid / 30) ** 2)).astype(dtype)
    fits, windows, counts = api.fit_spectra_batch(
        spectra=np.tile(signal, (4, 1)),
        dopp_slit=grid,
        spec_noise=np.ones((4, grid.size), dtype=np.float32),
        guide_velocities=0.0,
        meta=True,
        **COMMON_KW,
        **options,
    )
    return np.column_stack((fits, windows, counts))


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_call_and_scoped_workers_reach_the_kernel(monkeypatch, dtype):
    pools = []
    original = api._fit_spectra_batch  # noqa: SLF001

    def record(*args):
        pools.append(args[-1])
        return original(*args)

    monkeypatch.setattr(api, "_fit_spectra_batch", record)
    expected = run(dtype)

    def failing_stage():
        with set_workers(3):
            np.testing.assert_array_equal(run(dtype), expected)
            msg = "invalid stage"
            raise ValueError(msg)

    with set_workers(np.int32(2)):
        np.testing.assert_array_equal(run(dtype), expected)
        np.testing.assert_array_equal(run(dtype, workers=2), expected)
        np.testing.assert_array_equal(run(dtype, workers=1), expected)
        with pytest.raises(ValueError, match="invalid"):
            failing_stage()
        np.testing.assert_array_equal(run(dtype), expected)
    np.testing.assert_array_equal(run(dtype), expected)
    assert [p.num_threads if p is not None else None for p in pools] == [None, 2, 2, 1, 3, 2, None]
    assert pools[1] is pools[2] is pools[5]


def test_parallel_contexts_are_independent():
    rendezvous = Barrier(2)

    def stage(count):
        with set_workers(count):
            rendezvous.wait(timeout=10)
            return resolve_pool(None).num_threads

    with set_workers(3), ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(stage, [1, 2])) == [1, 2]
        assert resolve_pool(None).num_threads == 3


@pytest.mark.parametrize(
    ("value", "error"), [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_worker_counts(value, error):
    with pytest.raises(error):
        run(np.float32, workers=value)
    with pytest.raises(error), set_workers(value):
        pass
