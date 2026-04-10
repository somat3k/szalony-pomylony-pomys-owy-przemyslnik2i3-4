"""Plane E — Transformation Plane: tensorified modules.

A :class:`TransformModule` is the atomic unit of computation in the CROF
fabric.  Each module:

1. Declares its input schema (expected keys and types)
2. Accepts a typed :class:`~hololang.crof.envelope.Envelope`
3. Projects the payload into a working representation (tensor, embedding, dict)
4. Applies a deterministic or adaptive transformation
5. Emits a :class:`TransformResult` with the output, confidence, and provenance

A :class:`TransformGraph` chains modules as a DAG; the graph is executed
by :meth:`TransformGraph.run` which respects topological order.

Modules are classified by their *trust class*:

* ``deterministic`` — pure function; output is reproducible
* ``statistical``   — probabilistic; output may vary
* ``adaptive``      — parameters change during operation

Every transform is observable (logged to the :class:`~hololang.crof.audit.AuditLedger`
if one is wired in), composable (via :class:`TransformGraph`), and
attestable (via :class:`TransformResult.provenance`).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from hololang.tensor.tensor import Tensor
from hololang.tensor.embeddings import EmbeddingSpace
from hololang.crof.envelope import Envelope


# ---------------------------------------------------------------------------
# TransformResult
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """Output of a single :class:`TransformModule` execution.

    Attributes
    ----------
    module_id:
        Identifier of the module that produced this result.
    output:
        Primary output — may be a :class:`Tensor`, an :class:`EmbeddingSpace`,
        a Python scalar, a dict, or ``None``.
    confidence:
        Numeric confidence in [0, 1].  ``1.0`` = fully deterministic.
    provenance:
        List of input envelope IDs that contributed to this result.
    metadata:
        Arbitrary key-value extension fields.
    duration_ms:
        Wall-clock execution time in milliseconds.
    error:
        Non-empty string when the transform raised an exception.
    """
    module_id:   str            = ""
    output:      Any            = None
    confidence:  float          = 1.0
    provenance:  list[str]      = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)
    duration_ms: float          = 0.0
    error:       str            = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def __repr__(self) -> str:
        status = "OK" if self.ok else f"ERR:{self.error}"
        return (
            f"TransformResult(module={self.module_id!r}, "
            f"confidence={self.confidence:.2f}, {status})"
        )


# ---------------------------------------------------------------------------
# TransformModule  (Class-5 Tensor Compute Node)
# ---------------------------------------------------------------------------

class TransformModule:
    """A single tensorified transformation unit.

    Parameters
    ----------
    module_id:
        Unique identifier.
    fn:
        Callable ``(envelope, context) -> any`` that performs the transform.
        *context* is a free-form dict passed through the graph execution.
    trust_class:
        ``"deterministic"`` | ``"statistical"`` | ``"adaptive"``.
    description:
        Human-readable description of the transform.
    input_schema:
        Optional dict describing expected input keys and their types.
    output_schema:
        Optional dict describing the structure of the output.
    """

    def __init__(
        self,
        module_id: str = "",
        fn: Callable[[Envelope, dict[str, Any]], Any] | None = None,
        trust_class: str = "deterministic",
        description: str = "",
        input_schema:  dict[str, str] | None = None,
        output_schema: dict[str, str] | None = None,
    ) -> None:
        self.module_id     = module_id or str(uuid.uuid4())[:8]
        self.fn            = fn or (lambda env, ctx: None)
        self.trust_class   = trust_class
        self.description   = description
        self.input_schema  = input_schema  or {}
        self.output_schema = output_schema or {}
        self.invocations:  int   = 0
        self.total_ms:     float = 0.0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        envelope: Envelope,
        context:  dict[str, Any] | None = None,
    ) -> TransformResult:
        """Run the module's transform function.

        Parameters
        ----------
        envelope:
            The input envelope carrying the payload.
        context:
            Mutable execution context shared across a :class:`TransformGraph`
            run.

        Returns
        -------
        TransformResult
        """
        ctx = context or {}
        t0  = time.perf_counter()
        error = ""
        output = None
        try:
            output = self.fn(envelope, ctx)
        except Exception as exc:   # noqa: BLE001
            error = str(exc)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.invocations += 1
        self.total_ms    += dt_ms

        confidence = 1.0 if self.trust_class == "deterministic" else 0.9
        return TransformResult(
            module_id   = self.module_id,
            output      = output,
            confidence  = confidence,
            provenance  = [envelope.envelope_id],
            duration_ms = dt_ms,
            error       = error,
        )

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_ms / self.invocations) if self.invocations else 0.0

    def __repr__(self) -> str:
        return (
            f"TransformModule({self.module_id!r}, trust={self.trust_class!r}, "
            f"invocations={self.invocations})"
        )


# ---------------------------------------------------------------------------
# TransformGraph
# ---------------------------------------------------------------------------

@dataclass
class _TGNode:
    module:  TransformModule
    inputs:  list[str] = field(default_factory=list)   # module IDs of upstream nodes
    outputs: list[str] = field(default_factory=list)   # module IDs of downstream nodes


class TransformGraph:
    """A DAG of :class:`TransformModule` objects.

    Modules are connected with directed edges (producer → consumer).
    :meth:`run` executes them in topological order, passing each module's
    :class:`TransformResult` to downstream modules via the shared *context*.

    Usage::

        g = TransformGraph("pipeline")
        g.add(decode_module)
        g.add(embed_module)
        g.add(classify_module)
        g.connect("decode",   "embed")
        g.connect("embed",    "classify")
        results = g.run(envelope)
    """

    def __init__(self, name: str = "transform_graph") -> None:
        self.name   = name
        self._nodes: dict[str, _TGNode] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add(self, module: TransformModule) -> None:
        if module.module_id in self._nodes:
            raise ValueError(f"Module {module.module_id!r} already in graph")
        self._nodes[module.module_id] = _TGNode(module=module)

    def connect(self, src_id: str, dst_id: str) -> None:
        if src_id not in self._nodes:
            raise KeyError(f"Source module {src_id!r} not found")
        if dst_id not in self._nodes:
            raise KeyError(f"Destination module {dst_id!r} not found")
        self._nodes[src_id].outputs.append(dst_id)
        self._nodes[dst_id].inputs.append(src_id)

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topo(self) -> list[str]:
        visited: set[str] = set()
        order:   list[str] = []

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for inp in self._nodes[nid].inputs:
                dfs(inp)
            order.append(nid)

        for nid in self._nodes:
            dfs(nid)
        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        envelope: Envelope,
        context:  dict[str, Any] | None = None,
    ) -> dict[str, TransformResult]:
        """Execute the full transform graph.

        Returns a mapping of *module_id → TransformResult* for every module.
        Each result is also written to *context* under its module ID so that
        downstream modules can read upstream outputs.
        """
        ctx = context if context is not None else {}
        order   = self._topo()
        results: dict[str, TransformResult] = {}

        for nid in order:
            node   = self._nodes[nid]
            result = node.module.execute(envelope, ctx)
            results[nid] = result
            ctx[nid]     = result   # expose to downstream modules

        return results

    def __repr__(self) -> str:
        return f"TransformGraph(name={self.name!r}, modules={len(self._nodes)})"
