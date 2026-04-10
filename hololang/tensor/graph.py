"""Computation graph for lazy / deferred tensor evaluation.

Nodes represent operations; edges represent data flow.
The graph supports:
* Forward evaluation (topological sort)
* Gradient accumulation — reverse-mode autograd (backward pass)
* Batched execution via :class:`~hololang.tensor.pool.TensorPool`
"""

from __future__ import annotations

import math
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
        # Populated after backward pass
        self.grad:     Tensor | None = None
        # Vector-Jacobian product: (grad_output) -> list[grad_per_input]
        self.vjp:      Callable[[Tensor], list[Tensor]] | None = None

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
        grads   = g.backward()   # reverse-mode autograd
    """

    def __init__(self, name: str = "graph") -> None:
        self.name:   str = name
        self._nodes: dict[str, GraphNode] = {}
        self._order: list[GraphNode] | None = None

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def constant(self, tensor: Tensor, name: str = "") -> GraphNode:
        """Register a leaf (constant input) node — no gradient."""
        node = GraphNode(
            op="constant",
            inputs=[],
            fn=lambda: tensor,
            name=name or f"const_{len(self._nodes)}",
        )
        # Constants have no gradient
        node.vjp = None
        self._nodes[node.id] = node
        self._order = None   # invalidate cached topo order
        return node

    def variable(self, tensor: Tensor, name: str = "") -> GraphNode:
        """Register a trainable variable node — accumulates gradient."""
        node = GraphNode(
            op="variable",
            inputs=[],
            fn=lambda: tensor,
            name=name or f"var_{len(self._nodes)}",
        )
        # Variables have no upstream vjp but they accumulate incoming grad
        node.vjp = None
        self._nodes[node.id] = node
        self._order = None
        return node

    def op(
        self,
        op_name: str,
        inputs: list[GraphNode],
        fn: Callable[..., Tensor],
        name: str = "",
        vjp: Callable[[Tensor], list[Tensor]] | None = None,
    ) -> GraphNode:
        """Add a computation node with an optional vector-Jacobian product."""
        node = GraphNode(op=op_name, inputs=inputs, fn=fn, name=name)
        node.vjp = vjp
        self._nodes[node.id] = node
        self._order = None
        return node

    # ------------------------------------------------------------------
    # Convenience op builders (with autograd vjps)
    # ------------------------------------------------------------------

    def add(self, a: GraphNode, b: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            return [g, g]
        return self.op("add", [a, b], lambda x, y: x + y,
                       name=f"add({a.name},{b.name})", vjp=_vjp)

    def sub(self, a: GraphNode, b: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            return [g, -g]
        return self.op("sub", [a, b], lambda x, y: x - y,
                       name=f"sub({a.name},{b.name})", vjp=_vjp)

    def mul(self, a: GraphNode, b: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            assert a.output is not None and b.output is not None
            return [g * b.output, g * a.output]
        return self.op("mul", [a, b], lambda x, y: x * y,
                       name=f"mul({a.name},{b.name})", vjp=_vjp)

    def matmul(self, a: GraphNode, b: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            assert a.output is not None and b.output is not None
            grad_a = g.matmul(b.output.transpose())
            grad_b = a.output.transpose().matmul(g)
            return [grad_a, grad_b]
        return self.op("matmul", [a, b], lambda x, y: x.matmul(y),
                       name=f"matmul({a.name},{b.name})", vjp=_vjp)

    def relu(self, a: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            assert a.output is not None
            mask = a.output.apply_fn(lambda v: 1.0 if v > 0.0 else 0.0)
            return [g * mask]
        return self.op("relu", [a],
                       lambda x: x.apply_fn(lambda v: max(0.0, v)),
                       name=f"relu({a.name})", vjp=_vjp)

    def sigmoid(self, a: GraphNode) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            assert a.output is not None
            # sig * (1 - sig)
            node_out = a.output.apply_fn(lambda v: 1.0 / (1.0 + math.exp(-v)))
            one_minus = node_out.apply_fn(lambda v: 1.0 - v)
            return [g * node_out * one_minus]
        return self.op("sigmoid", [a],
                       lambda x: x.apply_fn(lambda v: 1 / (1 + math.exp(-v))),
                       name=f"sigmoid({a.name})", vjp=_vjp)

    def normalize(self, a: GraphNode) -> GraphNode:
        # Normalization backward is complex; mark as non-differentiable here
        return self.op("normalize", [a],
                       lambda x: x.normalize(),
                       name=f"normalize({a.name})", vjp=None)

    def scale(self, a: GraphNode, factor: float) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            return [g * factor]
        return self.op("scale", [a],
                       lambda x: x * factor,
                       name=f"scale({a.name},{factor})", vjp=_vjp)

    def reshape(self, a: GraphNode, *dims: int) -> GraphNode:
        def _vjp(g: Tensor) -> list[Tensor]:
            assert a.output is not None
            return [g.reshape(*a.output.dims)]
        return self.op("reshape", [a],
                       lambda x: x.reshape(*dims),
                       name=f"reshape({a.name},{dims})", vjp=_vjp)

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
            input_tensors = [inp.output for inp in node.inputs]
            missing = [
                inp.name for inp, t in zip(node.inputs, input_tensors) if t is None
            ]
            if missing:
                raise ValueError(
                    f"Cannot evaluate node {node.name!r}: missing output from "
                    f"upstream node(s): {', '.join(missing)}"
                )
            node.output = node.fn(*input_tensors)
            results[node.name] = node.output
        return results

    # ------------------------------------------------------------------
    # Backward pass (reverse-mode autograd)
    # ------------------------------------------------------------------

    def backward(
        self, output_grad: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Reverse-mode gradient accumulation.

        Must be called after :meth:`forward`.  Walks the computation graph in
        reverse topological order, accumulating gradients in each
        :attr:`GraphNode.grad` field.

        Parameters
        ----------
        output_grad:
            Seed gradient (upstream gradient for the terminal output node).
            When ``None`` a ones-tensor matching the last node's output shape
            is used, which is equivalent to computing ``d(sum(output)) / dx``
            for all inputs ``x``.

        Returns
        -------
        dict
            Mapping from node name to its accumulated gradient :class:`Tensor`.
        """
        order = self._topo_sort()
        if not order:
            return {}

        # Reset gradients
        for node in order:
            node.grad = None

        terminal = order[-1]
        if terminal.output is None:
            raise RuntimeError("forward() must be called before backward()")

        # Seed the terminal node's gradient
        if output_grad is None:
            output_grad = Tensor(
                terminal.output.dims,
                [1.0] * terminal.output.size,
                terminal.output.dtype,
            )
        terminal.grad = output_grad

        # Traverse in reverse
        for node in reversed(order):
            if node.grad is None or node.vjp is None or not node.inputs:
                continue
            upstream_grads = node.vjp(node.grad)
            for inp, g in zip(node.inputs, upstream_grads):
                if inp.grad is None:
                    inp.grad = g
                else:
                    inp.grad = inp.grad + g

        # Collect all non-None gradients
        return {
            node.name: node.grad
            for node in order
            if node.grad is not None
        }

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
