"""Computation graph for lazy / deferred tensor evaluation.

Nodes represent operations; edges represent data flow.
The graph supports:
* Forward evaluation (topological sort)
* Gradient accumulation (simple autograd for scalar losses)
* Batched execution via :class:`~hololang.tensor.pool.TensorPool`
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Computation graph
# ---------------------------------------------------------------------------

class ComputationGraph:
    """Directed Acyclic Graph of tensor operations.

    Usage::

        g = ComputationGraph()
        a = g.constant(Tensor.ones(3, 3), "A")
        b = g.constant(Tensor.eye(3),    "I")
        c = g.op("matmul", [a, b], fn=lambda x, y: x.matmul(y), name="C")
        outputs = g.forward()
    """

    def __init__(self, name: str = "graph") -> None:
        self.name:   str = name
        self._nodes: dict[str, GraphNode] = {}
        self._order: list[GraphNode] | None = None

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def constant(self, tensor: Tensor, name: str = "") -> GraphNode:
        """Register a leaf (constant input) node."""
        node = GraphNode(
            op="constant",
            inputs=[],
            fn=lambda: tensor,
            name=name or f"const_{len(self._nodes)}",
        )
        self._nodes[node.id] = node
        self._order = None   # invalidate cached topo order
        return node

    def variable(self, tensor: Tensor, name: str = "") -> GraphNode:
        """Register a trainable variable node."""
        node = GraphNode(
            op="variable",
            inputs=[],
            fn=lambda: tensor,
            name=name or f"var_{len(self._nodes)}",
        )
        self._nodes[node.id] = node
        self._order = None
        return node

    def op(
        self,
        op_name: str,
        inputs: list[GraphNode],
        fn: Callable[..., Tensor],
        name: str = "",
    ) -> GraphNode:
        """Add a computation node."""
        node = GraphNode(op=op_name, inputs=inputs, fn=fn, name=name)
        self._nodes[node.id] = node
        self._order = None
        return node

    # ------------------------------------------------------------------
    # Convenience op builders
    # ------------------------------------------------------------------

    def add(self, a: GraphNode, b: GraphNode) -> GraphNode:
        return self.op("add", [a, b], lambda x, y: x + y,
                       name=f"add({a.name},{b.name})")

    def sub(self, a: GraphNode, b: GraphNode) -> GraphNode:
        return self.op("sub", [a, b], lambda x, y: x - y,
                       name=f"sub({a.name},{b.name})")

    def mul(self, a: GraphNode, b: GraphNode) -> GraphNode:
        return self.op("mul", [a, b], lambda x, y: x * y,
                       name=f"mul({a.name},{b.name})")

    def matmul(self, a: GraphNode, b: GraphNode) -> GraphNode:
        return self.op("matmul", [a, b], lambda x, y: x.matmul(y),
                       name=f"matmul({a.name},{b.name})")

    def relu(self, a: GraphNode) -> GraphNode:
        return self.op("relu", [a],
                       lambda x: x.apply_fn(lambda v: max(0.0, v)),
                       name=f"relu({a.name})")

    def sigmoid(self, a: GraphNode) -> GraphNode:
        import math
        return self.op("sigmoid", [a],
                       lambda x: x.apply_fn(lambda v: 1 / (1 + math.exp(-v))),
                       name=f"sigmoid({a.name})")

    def normalize(self, a: GraphNode) -> GraphNode:
        return self.op("normalize", [a],
                       lambda x: x.normalize(),
                       name=f"normalize({a.name})")

    def scale(self, a: GraphNode, factor: float) -> GraphNode:
        return self.op("scale", [a],
                       lambda x: x * factor,
                       name=f"scale({a.name},{factor})")

    def reshape(self, a: GraphNode, *dims: int) -> GraphNode:
        return self.op("reshape", [a],
                       lambda x: x.reshape(*dims),
                       name=f"reshape({a.name},{dims})")

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topo_sort(self) -> list[GraphNode]:
        visited: set[str] = set()
        order:   list[GraphNode] = []

        def dfs(node: GraphNode) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for inp in node.inputs:
                dfs(inp)
            order.append(node)

        for node in self._nodes.values():
            dfs(node)

        self._order = order
        return order

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self) -> dict[str, Tensor]:
        """Evaluate all nodes in topological order.

        Returns
        -------
        dict
            Mapping from node name to its output :class:`Tensor`.
        """
        order = self._topo_sort()
        results: dict[str, Tensor] = {}
        for node in order:
            input_tensors = [
                inp.output for inp in node.inputs if inp.output is not None
            ]
            node.output = node.fn(*input_tensors)
            results[node.name] = node.output
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the graph."""
        lines = [f'digraph "{self.name}" {{', '  rankdir=LR;']
        for node in self._nodes.values():
            label = f"{node.op}\\n{node.name}"
            lines.append(f'  "{node.id}" [label="{label}"];')
        for node in self._nodes.values():
            for inp in node.inputs:
                lines.append(f'  "{inp.id}" -> "{node.id}";')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ComputationGraph(name={self.name!r}, nodes={len(self._nodes)})"
