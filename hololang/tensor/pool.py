"""Pooled tensor runtime.

Manages a collection of reusable :class:`~hololang.tensor.tensor.Tensor`
buffers and dispatches batch operations across them in a pool-of-workers
fashion (single-threaded by default; switch to ``concurrent.futures`` via
:meth:`TensorPool.set_parallel`).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# Pool entry
# ---------------------------------------------------------------------------

class _PoolEntry:
    def __init__(self, tensor: Tensor, tag: str) -> None:
        self.tensor    = tensor
        self.tag       = tag
        self.in_use    = False
        self.last_used = time.monotonic()


# ---------------------------------------------------------------------------
# TensorPool
# ---------------------------------------------------------------------------

class TensorPool:
    """A pool of pre-allocated tensor buffers.

    Parameters
    ----------
    max_workers:
        Maximum concurrent threads for parallel batch operations.
        Set to ``1`` (default) for deterministic sequential execution.
    """

    def __init__(self, name: str = "pool", max_workers: int = 1) -> None:
        self.name:  str = name
        self._pool: dict[str, _PoolEntry] = {}
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers = max_workers
        if max_workers > 1:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shutdown the thread-pool executor if running."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "TensorPool":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def allocate(self, tag: str, *dims: int,
                 dtype: str = "float32") -> Tensor:
        """Allocate a named tensor buffer in the pool."""
        with self._lock:
            if tag in self._pool:
                entry = self._pool[tag]
                if entry.tensor.shape == tuple(dims):
                    entry.in_use = False
                    return entry.tensor
            t = Tensor(list(dims), dtype=dtype, name=tag)
            self._pool[tag] = _PoolEntry(t, tag)
            return t

    def get(self, tag: str) -> Tensor | None:
        with self._lock:
            entry = self._pool.get(tag)
            return entry.tensor if entry else None

    def release(self, tag: str) -> None:
        with self._lock:
            if tag in self._pool:
                self._pool[tag].in_use = False

    def free(self, tag: str) -> None:
        with self._lock:
            self._pool.pop(tag, None)

    @property
    def tags(self) -> list[str]:
        with self._lock:
            return list(self._pool.keys())

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def map(
        self,
        fn: Callable[[Tensor], Tensor],
        tags: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Apply *fn* to each pooled tensor (or a subset by *tags*).

        Returns
        -------
        dict
            ``{tag: result_tensor}``
        """
        targets = tags or self.tags

        if self._executor is not None:
            futures: dict[str, Future] = {}
            for tag in targets:
                t = self.get(tag)
                if t is not None:
                    futures[tag] = self._executor.submit(fn, t)
            return {tag: f.result() for tag, f in futures.items()}

        # Sequential fallback
        results: dict[str, Tensor] = {}
        for tag in targets:
            t = self.get(tag)
            if t is not None:
                results[tag] = fn(t)
        return results

    def reduce(
        self,
        fn: Callable[[Tensor, Tensor], Tensor],
        tags: list[str] | None = None,
        initial: Tensor | None = None,
    ) -> Tensor | None:
        """Reduce all pooled tensors with *fn*.

        Parameters
        ----------
        fn:
            Binary function ``(accumulator, tensor) -> tensor``.
        initial:
            Starting accumulator.  If ``None`` the first tensor is used.
        """
        targets = tags or self.tags
        acc = initial
        for tag in targets:
            t = self.get(tag)
            if t is None:
                continue
            acc = fn(acc, t) if acc is not None else t
        return acc

    # ------------------------------------------------------------------
    # Polygon-range dimensional surface processing
    # ------------------------------------------------------------------

    def process_surface(
        self,
        tag: str,
        polygon_fn: Callable[[int, int, float], float],
    ) -> Tensor:
        """Apply *polygon_fn* to every element of a 2-D surface tensor.

        Parameters
        ----------
        tag:
            Pool tag for the tensor.
        polygon_fn:
            ``(row, col, current_value) -> new_value``
        """
        t = self.get(tag)
        if t is None:
            raise KeyError(f"No tensor with tag {tag!r} in pool")
        if t.ndim != 2:
            raise ValueError("process_surface() requires a 2-D tensor")
        rows, cols = t.dims
        new_data = [
            polygon_fn(r, c, t._data[r * cols + c])
            for r in range(rows)
            for c in range(cols)
        ]
        result = Tensor([rows, cols], new_data, t.dtype, t.name)
        with self._lock:
            self._pool[tag].tensor = result
        return result

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TensorPool(name={self.name!r}, "
            f"buffers={len(self._pool)}, "
            f"workers={self._max_workers})"
        )

    def summary(self) -> str:
        lines = [f"TensorPool '{self.name}':"]
        for tag, entry in self._pool.items():
            lines.append(
                f"  [{tag}]  shape={entry.tensor.shape}  "
                f"dtype={entry.tensor.dtype}  in_use={entry.in_use}"
            )
        return "\n".join(lines)
