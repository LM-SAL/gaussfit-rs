"""
Control the Rayon workers used for parallel calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from operator import index
from typing import TYPE_CHECKING

import numpy as np

from ._gaussfit_rs import _WorkerPool

if TYPE_CHECKING:
    from collections.abc import Iterator

_active: ContextVar[_WorkerPool | None] = ContextVar("gaussfit_rs_pool", default=None)


def _pool_for_count(workers: int) -> _WorkerPool:
    if isinstance(workers, (bool, np.bool_)):
        msg = "workers must be a positive integer"
        raise TypeError(msg)
    count = index(workers)
    if count < 1:
        msg = "workers must be a positive integer"
        raise ValueError(msg)
    active = _active.get()
    if active is not None and active.num_threads == count:
        return active
    return _WorkerPool(count)


def resolve_pool(workers: int | None) -> _WorkerPool | None:
    """
    Select the explicit pool, or inherit the current context.
    """
    return _active.get() if workers is None else _pool_for_count(workers)


@contextmanager
def set_workers(workers: int) -> Iterator[None]:
    """
    Reuse a pool of ``workers`` threads for parallel calls in this context.

    An explicit ``workers=`` on a call overrides the context. Leaving the context restores the
    previous pool, including when a call raises.
    """
    token = _active.set(_pool_for_count(workers))
    try:
        yield
    finally:
        _active.reset(token)
