"""SpreadTensor and BatchContainer — sharding for the tensor subsystem.

:class:`SpreadTensor` is a view over a sub-region of a parent
:class:`~hololang.tensor.tensor.Tensor`, describing one shard.  A
:class:`BatchContainer` holds an ordered collection of :class:`SpreadTensor`
shards and can scatter work across multiple CPU threads via
:class:`~concurrent.futures.ThreadPoolExecutor`.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# SpreadTensor — a shard window into a parent tensor
# ---------------------------------------------------------------------------

class SpreadTensor:
    """A contiguous sub-region (shard) of a parent :class:`Tensor`.

    Parameters
    ----------
    parent:
        The source tensor being sharded.
    offset:
        Flat-data index where this shard begins.
    length:
        Number of elements in this shard.
    shard_index:
        Zero-based index of this shard within its :class:`BatchContainer`.
    meta:
        Arbitrary key/value metadata (e.g. node id, generation counter).
    """

    def __init__(
        self,
        parent: Tensor,
        offset: int,
        length: int,
        shard_index: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.id          = str(uuid.uuid4())[:8]
        self.parent      = parent
        self.offset      = offset
        self.length      = length
        self.shard_index = shard_index
        self.meta: dict[str, Any] = meta or {}

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def gather(self) -> Tensor:
        """Return a 1-D :class:`Tensor` containing the shard's elements."""
        data = self.parent._data[self.offset: self.offset + self.length]
        return Tensor([self.length], list(data), self.parent.dtype,
                      name=f"shard_{self.shard_index}")

    def scatter(self, t: Tensor) -> None:
        """Write *t*'s data back into the parent tensor at this shard's window.

        Parameters
        ----------
        t:
            Tensor whose size must equal :attr:`length`.
        """
        if t.size != self.length:
            raise ValueError(
                f"SpreadTensor.scatter: expected {self.length} elements, "
                f"got {t.size}"
            )
        self.parent._data[self.offset: self.offset + self.length] = list(t._data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Gather, apply *fn*, scatter back, and return the result."""
        result = fn(self.gather())
        self.scatter(result)
        return result

    def __repr__(self) -> str:
        return (
            f"SpreadTensor(id={self.id!r}, shard={self.shard_index}, "
            f"offset={self.offset}, length={self.length})"
        )


# ---------------------------------------------------------------------------
# BatchContainer — ordered collection of SpreadTensor shards
# ---------------------------------------------------------------------------

class BatchContainer:
    """A batch of :class:`SpreadTensor` shards derived from one parent tensor.

    Parameters
    ----------
    name:
        Identifier for this batch.
    parent:
        The source tensor that was split.
    shards:
        Pre-built list of :class:`SpreadTensor` instances.
    meta:
        Arbitrary batch-level metadata.
    """

    def __init__(
        self,
        name: str = "batch",
        parent: Tensor | None = None,
        shards: list[SpreadTensor] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.id     = str(uuid.uuid4())[:8]
        self.name   = name
        self.parent = parent
        self.shards: list[SpreadTensor] = shards or []
        self.meta: dict[str, Any] = meta or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_tensor(
        cls,
        parent: Tensor,
        n_shards: int,
        name: str = "batch",
    ) -> "BatchContainer":
        """Split *parent* into *n_shards* equal (or near-equal) shards.

        Parameters
        ----------
        parent:
            Tensor to shard.  Its flat data is split contiguously.
        n_shards:
            Number of shards; each shard carries ``ceil(size / n_shards)``
            elements (last shard may be smaller).
        """
        if n_shards < 1:
            raise ValueError("n_shards must be >= 1")
        total = parent.size
        base  = total // n_shards
        extra = total % n_shards
        shards: list[SpreadTensor] = []
        offset = 0
        for i in range(n_shards):
            length = base + (1 if i < extra else 0)
            if length == 0:
                break
            shards.append(SpreadTensor(parent, offset, length, i))
            offset += length
        return cls(name=name, parent=parent, shards=shards)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def apply(
        self,
        fn: Callable[[Tensor], Tensor],
        workers: int = 1,
    ) -> list[Tensor]:
        """Apply *fn* to every shard (optionally in parallel).

        Parameters
        ----------
        fn:
            Callable ``(Tensor) -> Tensor`` applied to each shard's gathered
            data.  Results are scattered back into the parent tensor.
        workers:
            Number of threads.  ``1`` = sequential (deterministic order).

        Returns
        -------
        list[Tensor]
            Result tensor for each shard in order.
        """
        if workers <= 1:
            return [s.apply(fn) for s in self.shards]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures: list[Future] = [
                ex.submit(s.apply, fn) for s in self.shards
            ]
        return [f.result() for f in futures]

    def gather_all(self) -> Tensor:
        """Reconstruct a single 1-D :class:`Tensor` from all shards."""
        data: list[float] = []
        for s in self.shards:
            data.extend(s.gather()._data)
        return Tensor([len(data)], data, self.parent.dtype if self.parent else "float32")

    def reduce(
        self,
        fn: Callable[[Tensor, Tensor], Tensor],
        initial: Tensor | None = None,
    ) -> Tensor | None:
        """Reduce all shard tensors with *fn*.

        Parameters
        ----------
        fn:
            Binary function ``(accumulator, shard_tensor) -> tensor``.
        initial:
            Starting value; if *None* the first shard is used.
        """
        acc = initial
        for s in self.shards:
            t = s.gather()
            acc = fn(acc, t) if acc is not None else t
        return acc

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self.shards)

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> SpreadTensor:
        return self.shards[i]

    def __repr__(self) -> str:
        return (
            f"BatchContainer(id={self.id!r}, name={self.name!r}, "
            f"shards={len(self.shards)})"
        )
