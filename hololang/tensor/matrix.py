"""CPU-threaded matrix engine and dynamic parameter multiplier.

:class:`MatrixEngine` wraps the pure-Python matmul of
:class:`~hololang.tensor.tensor.Tensor` with a
:class:`~concurrent.futures.ThreadPoolExecutor` to parallelise row-block
computation across all available CPU cores.

:class:`ParameterMultiplier` provides a dynamic scalar multiplier that can
be applied to any tensor hyperparameter, supporting per-step or per-epoch
scaling policies (warmup, decay, cyclic, step).
"""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# MatrixEngine — threaded matmul
# ---------------------------------------------------------------------------

class MatrixEngine:
    """CPU multi-threaded matrix multiplication engine.

    Parameters
    ----------
    workers:
        Number of parallel threads.  Defaults to ``os.cpu_count()`` or 4.
    block_rows:
        Rows per thread block.  Smaller values increase parallelism but
        add scheduling overhead; tune for your workload.
    """

    def __init__(self, workers: int | None = None, block_rows: int = 16) -> None:
        import os
        self._workers = workers or (os.cpu_count() or 4)
        self._block_rows = max(1, block_rows)
        self._executor = ThreadPoolExecutor(max_workers=self._workers)

    # ------------------------------------------------------------------
    # Internal serial matmul (no thread pool)
    # ------------------------------------------------------------------

    def _matmul_serial(self, a: Tensor, b: Tensor) -> Tensor:
        """Single-threaded matmul — safe to call from worker threads."""
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("MatrixEngine.matmul requires 2-D tensors")
        m, k = a.dims
        k2, n = b.dims
        if k != k2:
            raise ValueError(
                f"Incompatible shapes: {a.shape} @ {b.shape}"
            )
        result = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += a._data[i * k + p] * b._data[p * n + j]
                result[i * n + j] = s
        return Tensor([m, n], result, a.dtype)

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute ``a @ b`` using a parallel row-block strategy.

        Parameters
        ----------
        a:
            Shape ``(M, K)``.
        b:
            Shape ``(K, N)``.

        Returns
        -------
        Tensor
            Shape ``(M, N)``.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("MatrixEngine.matmul requires 2-D tensors")
        m, k = a.dims
        k2, n = b.dims
        if k != k2:
            raise ValueError(
                f"Incompatible shapes: {a.shape} @ {b.shape}"
            )

        result_data = [0.0] * (m * n)

        def _compute_rows(row_start: int, row_end: int) -> None:
            for i in range(row_start, row_end):
                for j in range(n):
                    s = 0.0
                    for p in range(k):
                        s += a._data[i * k + p] * b._data[p * n + j]
                    result_data[i * n + j] = s

        futures: list[Future] = []
        row = 0
        while row < m:
            end = min(row + self._block_rows, m)
            futures.append(self._executor.submit(_compute_rows, row, end))
            row = end

        for f in futures:
            f.result()

        return Tensor([m, n], result_data, a.dtype)

    # ------------------------------------------------------------------
    # Batch matmul — list of (A, B) pairs in parallel
    # ------------------------------------------------------------------

    def batch_matmul(
        self, pairs: list[tuple[Tensor, Tensor]]
    ) -> list[Tensor]:
        """Multiply each ``(A, B)`` pair in *pairs* concurrently.

        Each pair is evaluated with a serial (non-threaded) matmul so that
        worker threads do not re-enter the shared executor (which would
        deadlock).

        Returns
        -------
        list[Tensor]
            Results in the same order as *pairs*.
        """
        futures = [
            self._executor.submit(self._matmul_serial, a, b)
            for a, b in pairs
        ]
        return [f.result() for f in futures]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "MatrixEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"MatrixEngine(workers={self._workers}, "
            f"block_rows={self._block_rows})"
        )


# ---------------------------------------------------------------------------
# ParameterMultiplier — dynamic scalar scheduling
# ---------------------------------------------------------------------------

class ParameterMultiplier:
    """Dynamic scalar multiplier for tensor hyperparameters.

    Supports four scheduling policies:

    * ``"constant"``  — always returns *base_value*.
    * ``"warmup"``    — linearly ramps from 0 to *base_value* over
      *warmup_steps* steps, then stays constant.
    * ``"decay"``     — exponentially decays:
      ``base_value * decay_rate ** step``.
    * ``"cyclic"``    — cosine annealing between *min_value* and *base_value*
      with period *cycle_steps*.
    * ``"step"``      — multiplies by *decay_rate* every *step_size* steps.

    Parameters
    ----------
    base_value:
        The nominal (peak) value of the parameter.
    policy:
        One of ``"constant"``, ``"warmup"``, ``"decay"``, ``"cyclic"``,
        ``"step"``.
    min_value:
        Floor value (used by ``"cyclic"`` and ``"decay"``).
    warmup_steps:
        Ramp length for ``"warmup"`` policy.
    decay_rate:
        Multiplicative decay factor per step (``"decay"`` / ``"step"``).
    cycle_steps:
        Full-cycle length for ``"cyclic"`` policy.
    step_size:
        Steps between multiplications for ``"step"`` policy.
    """

    _POLICIES = frozenset({"constant", "warmup", "decay", "cyclic", "step"})

    def __init__(
        self,
        base_value: float = 1.0,
        policy: str = "constant",
        min_value: float = 0.0,
        warmup_steps: int = 100,
        decay_rate: float = 0.99,
        cycle_steps: int = 200,
        step_size: int = 100,
    ) -> None:
        if policy not in self._POLICIES:
            raise ValueError(
                f"Unknown policy {policy!r}. "
                f"Choose from: {sorted(self._POLICIES)}"
            )
        self.base_value   = base_value
        self.policy       = policy
        self.min_value    = min_value
        self.warmup_steps = max(1, warmup_steps)
        self.decay_rate   = decay_rate
        self.cycle_steps  = max(1, cycle_steps)
        self.step_size    = max(1, step_size)
        self._step        = 0

    def value(self, step: int | None = None) -> float:
        """Return the multiplier value at *step* (or the internal counter)."""
        s = step if step is not None else self._step
        if self.policy == "constant":
            return self.base_value
        if self.policy == "warmup":
            if s < self.warmup_steps:
                return self.base_value * s / self.warmup_steps
            return self.base_value
        if self.policy == "decay":
            return max(self.min_value, self.base_value * (self.decay_rate ** s))
        if self.policy == "cyclic":
            t = (s % self.cycle_steps) / self.cycle_steps
            cos = (1.0 + math.cos(math.pi * t)) / 2.0
            return self.min_value + (self.base_value - self.min_value) * cos
        if self.policy == "step":
            n_decays = s // self.step_size
            return max(
                self.min_value,
                self.base_value * (self.decay_rate ** n_decays),
            )
        return self.base_value  # fallback

    def step(self) -> float:
        """Return the value at the current step, then advance the internal counter."""
        v = self.value(self._step)
        self._step += 1
        return v

    def apply(self, t: Tensor, step: int | None = None) -> Tensor:
        """Scale every element of *t* by the current multiplier value."""
        return t * self.value(step)

    def reset(self) -> None:
        """Reset the internal step counter to zero."""
        self._step = 0

    def __repr__(self) -> str:
        return (
            f"ParameterMultiplier(policy={self.policy!r}, "
            f"base={self.base_value}, step={self._step})"
        )
