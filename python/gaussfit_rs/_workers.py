"""
Control the Rayon workers used for parallel calls.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from operator import index
from typing import TYPE_CHECKING

import numpy as np

from ._gaussfit_rs import _WorkerPool

if TYPE_CHECKING:
    from collections.abc import Iterator

_active: ContextVar[_WorkerPool | None] = ContextVar("gaussfit_rs_pool", default=None)


# Fork copies Rayon's registry without its threads. Import alone does not initialise Rayon.
_pool_pid = cache(os.getpid)


def _active_pool() -> _WorkerPool | None:
    if _pool_pid() != os.getpid():
        msg = "Rayon workers were used in the parent process. Use 'spawn' or 'forkserver'."
        raise RuntimeError(msg)
    return _active.get()


def _pool_for_count(workers: int) -> _WorkerPool:
    if isinstance(workers, (bool, np.bool_)):
        msg = "workers must be a positive integer"
        raise TypeError(msg)
    count = index(workers)
    if count < 1:
        msg = "workers must be a positive integer"
        raise ValueError(msg)
    active = _active_pool()
    if active is not None and active.num_threads == count:
        return active
    return _WorkerPool(count)


def resolve_pool(workers: int | None) -> _WorkerPool | None:
    """
    Use the requested thread count or inherit the active pool.
    """
    return _active_pool() if workers is None else _pool_for_count(workers)


@contextmanager
def set_workers(workers: int) -> Iterator[None]:
    """
    Reuse a pool of ``workers`` threads for parallel calls in this context.

    A call can override the count with ``workers``. Exiting the block restores the previous pool.
    """
    token = _active.set(_pool_for_count(workers))
    try:
        yield
    finally:
        _active.reset(token)
