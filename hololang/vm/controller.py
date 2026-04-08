"""Block controller generative engine.

The :class:`BlockController` manages a directed graph of processing blocks
where each block is a named callable that can be connected, composed, and
parameterised.  It drives the generative / directive processing loop used
by the HoloLang MDI canvas.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Processing block
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """A named processing unit in the block graph.

    Parameters
    ----------
    name:
        Unique name within the controller.
    fn:
        ``(*inputs, **params) -> output``  callable.
    params:
        Static parameters merged into every ``fn`` call.
    """

    name:    str
    fn:      Callable
    params:  dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    _output: Any = field(default=None, init=False, repr=False)

    def execute(self, *inputs: Any) -> Any:
        if not self.enabled:
            return inputs[0] if inputs else None
        self._output = self.fn(*inputs, **self.params)
        return self._output

    @property
    def last_output(self) -> Any:
        return self._output


# ---------------------------------------------------------------------------
# BlockController
# ---------------------------------------------------------------------------

class BlockController:
    """Generative engine composed of interconnected processing blocks.

    Connections form a directed graph; execution follows topological order.
    Supports:
    * Named blocks with arbitrary Python callables
    * Parameter injection per block
    * Enable / disable individual blocks
    * Iterative generation (run N times, streaming output)

    Example::

        bc = BlockController("pipeline")
        bc.add_block("scale",  fn=lambda x: x * 2.0)
        bc.add_block("offset", fn=lambda x, bias=0: x + bias, bias=1.0)
        bc.connect("scale", "offset")
        result = bc.run(5.0)   # → 11.0
    """

    def __init__(self, name: str = "controller") -> None:
        self.name:   str = name
        self._blocks: dict[str, Block] = {}
        self._edges:  dict[str, list[str]] = {}   # src -> [dst]
        self._inputs: list[str] = []              # entry blocks (no incoming edges)
        self._log:    list[str] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_block(
        self,
        name: str,
        fn: Callable,
        **params: Any,
    ) -> "BlockController":
        self._blocks[name] = Block(name=name, fn=fn, params=params)
        self._edges.setdefault(name, [])
        return self

    def connect(self, src: str, dst: str) -> "BlockController":
        """Add a directed edge from *src* to *dst*."""
        if src not in self._blocks:
            raise KeyError(f"Unknown block {src!r}")
        if dst not in self._blocks:
            raise KeyError(f"Unknown block {dst!r}")
        self._edges[src].append(dst)
        return self

    def set_param(self, block_name: str, **kwargs: Any) -> None:
        self._blocks[block_name].params.update(kwargs)

    def enable(self, block_name: str) -> None:
        self._blocks[block_name].enabled = True

    def disable(self, block_name: str) -> None:
        self._blocks[block_name].enabled = False

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topo_order(self) -> list[str]:
        # Kahn's algorithm
        in_degree: dict[str, int] = {name: 0 for name in self._blocks}
        for src, dsts in self._edges.items():
            for dst in dsts:
                in_degree[dst] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dst in self._edges.get(node, []):
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)
        if len(order) != len(self._blocks):
            raise RuntimeError("BlockController: cycle detected in block graph")
        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_input: Any = None) -> Any:
        """Execute all blocks in topological order.

        Parameters
        ----------
        initial_input:
            Value fed to the first block(s).

        Returns
        -------
        Any
            Output of the last block in execution order.
        """
        order = self._topo_order()
        values: dict[str, Any] = {}
        result: Any = None

        for name in order:
            block = self._blocks[name]
            # Gather inputs from predecessor blocks
            predecessors = [
                src for src, dsts in self._edges.items() if name in dsts
            ]
            if predecessors:
                inputs = [values[p] for p in predecessors if p in values]
            else:
                inputs = [initial_input] if initial_input is not None else []

            values[name] = block.execute(*inputs)
            result = values[name]
            self._log.append(f"[{name}] output={result!r}")

        return result

    def generate(
        self,
        seed: Any = None,
        iterations: int = 1,
    ) -> Iterator[Any]:
        """Repeatedly run the pipeline, feeding output back as input.

        Yields the output of each iteration.
        """
        current = seed
        for _ in range(iterations):
            current = self.run(current)
            yield current

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation."""
        lines = [f'digraph "{self.name}" {{', '  rankdir=LR;']
        for name in self._blocks:
            b = self._blocks[name]
            style = "" if b.enabled else ' style="dashed" color="gray"'
            lines.append(f'  "{name}" [label="{name}"{style}];')
        for src, dsts in self._edges.items():
            for dst in dsts:
                lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)

    def get_log(self) -> list[str]:
        return list(self._log)

    def __repr__(self) -> str:
        return (
            f"BlockController(name={self.name!r}, "
            f"blocks={len(self._blocks)}, "
            f"edges={sum(len(v) for v in self._edges.values())})"
        )
