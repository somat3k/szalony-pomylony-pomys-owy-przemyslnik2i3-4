"""Pooled VM runtime – manages a pool of :class:`~hololang.vm.kernel.Kernel`
instances, scheduling work across them and orchestrating the
:class:`~hololang.vm.controller.BlockController` generative engine.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from hololang.vm.kernel import Kernel, KernelState
from hololang.vm.controller import BlockController


class PoolRuntime:
    """Manages a pool of replicable kernels with optional parallelism.

    Parameters
    ----------
    name:
        Pool identifier.
    max_kernels:
        Maximum simultaneous kernels (hard limit on pool size).
    max_workers:
        Thread-pool workers for parallel kernel execution (1 = serial).
    """

    def __init__(
        self,
        name: str = "runtime",
        max_kernels: int = 64,
        max_workers: int = 1,
    ) -> None:
        self.name:        str = name
        self.max_kernels: int = max_kernels
        self._kernels:    dict[str, Kernel] = {}
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        if max_workers > 1:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._controller: BlockController | None = None

    # ------------------------------------------------------------------
    # Kernel management
    # ------------------------------------------------------------------

    def create_kernel(self, name: str = "", **params: Any) -> Kernel:
        """Create and register a new kernel."""
        with self._lock:
            if len(self._kernels) >= self.max_kernels:
                raise RuntimeError(
                    f"PoolRuntime {self.name!r}: max_kernels ({self.max_kernels}) reached"
                )
            k = Kernel(name=name or f"k_{uuid.uuid4().hex[:6]}")
            k.params.update(params)
            self._kernels[k.name] = k
            return k

    def get_kernel(self, name: str) -> Kernel | None:
        return self._kernels.get(name)

    def replicate_kernel(self, source_name: str, copies: int = 1) -> list[Kernel]:
        """Replicate an existing kernel *copies* times."""
        source = self._kernels.get(source_name)
        if source is None:
            raise KeyError(f"Kernel {source_name!r} not found")
        clones: list[Kernel] = []
        for i in range(copies):
            clone = source.replicate(f"{source_name}_r{i}")
            with self._lock:
                self._kernels[clone.name] = clone
            clones.append(clone)
        return clones

    def remove_kernel(self, name: str) -> None:
        with self._lock:
            self._kernels.pop(name, None)

    @property
    def kernel_names(self) -> list[str]:
        with self._lock:
            return list(self._kernels.keys())

    # ------------------------------------------------------------------
    # Block controller
    # ------------------------------------------------------------------

    def set_controller(self, controller: BlockController) -> None:
        self._controller = controller

    def get_controller(self) -> BlockController | None:
        return self._controller

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_kernel(self, name: str, max_cycles: int = 100_000) -> Any:
        """Run a single kernel synchronously."""
        k = self._kernels.get(name)
        if k is None:
            raise KeyError(f"Kernel {name!r} not found")
        return k.run(max_cycles=max_cycles)

    def run_all(self, max_cycles: int = 100_000) -> dict[str, Any]:
        """Run all kernels, optionally in parallel.

        Returns
        -------
        dict
            ``{kernel_name: result}``
        """
        names = self.kernel_names
        if self._executor is not None:
            futures: dict[str, Future] = {
                n: self._executor.submit(self._kernels[n].run, max_cycles)
                for n in names
            }
            return {n: f.result() for n, f in futures.items()}
        return {n: self._kernels[n].run(max_cycles) for n in names}

    def run_pipeline(self, initial: Any = None,
                     iterations: int = 1) -> list[Any]:
        """Run the attached :class:`BlockController` generatively."""
        if self._controller is None:
            raise RuntimeError(
                f"PoolRuntime {self.name!r}: no controller attached"
            )
        return list(self._controller.generate(initial, iterations))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "PoolRuntime":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"PoolRuntime '{self.name}':"]
        for name, k in self._kernels.items():
            lines.append(f"  [{name}]  state={k.state.value}  "
                         f"stack={len(k._stack)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PoolRuntime(name={self.name!r}, "
            f"kernels={len(self._kernels)}/{self.max_kernels})"
        )
