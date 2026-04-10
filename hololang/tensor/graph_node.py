"""GraphNode — a single node in a tensor computation graph.

Extracted from :mod:`hololang.tensor.graph` so that node-level code can be
imported independently without pulling in the full
:class:`~hololang.tensor.graph.ComputationGraph`.
"""

from __future__ import annotations

import uuid
from typing import Callable

from hololang.tensor.tensor import Tensor


class GraphNode:
    """A single node in a computation graph.

    Parameters
    ----------
    op:
        Name of the operation (e.g. ``"matmul"``, ``"add"``, ``"relu"``).
    inputs:
        List of parent :class:`GraphNode` objects.
    fn:
        Python callable ``(*tensors) -> Tensor`` that performs the op.
    name:
        Human-readable label for debugging / visualisation.
    """

    def __init__(
        self,
        op: str,
        inputs: list["GraphNode"],
        fn: Callable[..., Tensor],
        name: str = "",
    ) -> None:
        self.id:       str = str(uuid.uuid4())[:8]
        self.op:       str = op
        self.inputs:   list["GraphNode"] = inputs
        self.fn:       Callable[..., Tensor] = fn
        self.name:     str = name or f"{op}_{self.id}"

        # Populated after forward pass
        self.output:   Tensor | None = None
        self.grad:     Tensor | None = None

    def __repr__(self) -> str:
        return f"GraphNode(op={self.op!r}, name={self.name!r})"
