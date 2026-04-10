"""Data transformation pipeline for the tensor subsystem.

Provides a composable, type-safe pipeline for applying sequences of tensor
operations.  Each :class:`Transform` is a callable that maps one
:class:`~hololang.tensor.tensor.Tensor` to another.

Usage::

    from hololang.tensor.transforms import TransformChain
    from hololang.tensor.ops import relu, layer_norm, min_max_scale

    pipeline = (
        TransformChain("preprocess")
        .add(layer_norm)
        .add(relu)
        .add(min_max_scale)
    )
    output = pipeline(input_tensor)
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterator

from hololang.tensor.tensor import Tensor

# A Transform is simply a callable from Tensor → Tensor.
Transform = Callable[[Tensor], Tensor]


# ---------------------------------------------------------------------------
# TransformChain
# ---------------------------------------------------------------------------

class TransformChain:
    """Ordered sequence of :data:`Transform` callables applied left-to-right.

    Parameters
    ----------
    name:
        Human-readable chain identifier.
    """

    def __init__(self, name: str = "chain") -> None:
        self.name: str = name
        self._transforms: list[tuple[str, Transform]] = []

    # ------------------------------------------------------------------
    # Building the chain
    # ------------------------------------------------------------------

    def add(self, fn: Transform, label: str = "") -> "TransformChain":
        """Append *fn* to the end of the chain.

        Parameters
        ----------
        fn:
            Callable ``(Tensor) -> Tensor``.
        label:
            Optional descriptive name for logging / debugging.

        Returns
        -------
        TransformChain
            *self* — enables fluent chaining.
        """
        name = label or getattr(fn, "__name__", repr(fn))
        self._transforms.append((name, fn))
        return self

    def prepend(self, fn: Transform, label: str = "") -> "TransformChain":
        """Insert *fn* at the beginning of the chain."""
        name = label or getattr(fn, "__name__", repr(fn))
        self._transforms.insert(0, (name, fn))
        return self

    def remove(self, label: str) -> "TransformChain":
        """Remove the first transform whose label matches *label*."""
        self._transforms = [
            (n, f) for n, f in self._transforms if n != label
        ]
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def __call__(self, t: Tensor) -> Tensor:
        """Run all transforms sequentially, returning the final tensor."""
        for _name, fn in self._transforms:
            t = fn(t)
        return t

    def trace(self, t: Tensor) -> list[tuple[str, Tensor]]:
        """Execute the chain and record the output after every step.

        Returns
        -------
        list of (label, Tensor)
            Intermediate outputs in order.
        """
        results: list[tuple[str, Tensor]] = []
        for name, fn in self._transforms:
            t = fn(t)
            results.append((name, t))
        return results

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def then(self, other: "TransformChain") -> "TransformChain":
        """Return a new chain that is *self* followed by *other*."""
        merged = TransformChain(f"{self.name}+{other.name}")
        merged._transforms = list(self._transforms) + list(other._transforms)
        return merged

    def __iter__(self) -> Iterator[tuple[str, Transform]]:
        return iter(self._transforms)

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:
        steps = [n for n, _ in self._transforms]
        return f"TransformChain(name={self.name!r}, steps={steps})"


# ---------------------------------------------------------------------------
# ParallelTransformGroup — apply multiple independent transforms to one tensor
# ---------------------------------------------------------------------------

class ParallelTransformGroup:
    """Apply several independent transforms to the same tensor concurrently.

    Each transform in the group runs in its own thread; results are returned
    as a dict keyed by label.

    Parameters
    ----------
    workers:
        Thread-pool size.  Defaults to the number of registered transforms.
    """

    def __init__(self, name: str = "parallel_group",
                 workers: int | None = None) -> None:
        self.name = name
        self._branches: list[tuple[str, Transform]] = []
        self._workers = workers
        self._lock = threading.Lock()

    def add(self, fn: Transform, label: str = "") -> "ParallelTransformGroup":
        name = label or getattr(fn, "__name__", repr(fn))
        self._branches.append((name, fn))
        return self

    def __call__(self, t: Tensor) -> dict[str, Tensor]:
        """Run all transforms concurrently.

        Returns
        -------
        dict[str, Tensor]
            ``{label: result}`` for each registered branch.
        """
        workers = self._workers or max(1, len(self._branches))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures: dict[str, Future] = {
                name: ex.submit(fn, t)
                for name, fn in self._branches
            }
        return {name: f.result() for name, f in futures.items()}

    def __repr__(self) -> str:
        labels = [n for n, _ in self._branches]
        return f"ParallelTransformGroup(name={self.name!r}, branches={labels})"
