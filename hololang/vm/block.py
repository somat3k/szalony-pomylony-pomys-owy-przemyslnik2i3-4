"""Block — a named processing unit for the BlockController generative engine.

Extracted from :mod:`hololang.vm.controller` so that blocks can be imported
and constructed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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
    enabled:
        When ``False`` the block passes its first input through unchanged.
    """

    name:    str
    fn:      Callable
    params:  dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    _output: Any = field(default=None, init=False, repr=False)

    def execute(self, *inputs: Any) -> Any:
        """Execute the block's function with *inputs* and stored params."""
        if not self.enabled:
            return inputs[0] if inputs else None
        self._output = self.fn(*inputs, **self.params)
        return self._output

    @property
    def last_output(self) -> Any:
        """The most recent output value, or ``None`` before first execution."""
        return self._output
