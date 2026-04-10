"""KernelState — lifecycle state enumeration for HoloLang VM kernels.

Extracted from :mod:`hololang.vm.kernel` so that external code can import the
enum without pulling in the full kernel implementation.
"""

from __future__ import annotations

from enum import Enum


class KernelState(Enum):
    """Lifecycle states of a :class:`~hololang.vm.kernel.Kernel`."""

    IDLE      = "idle"
    RUNNING   = "running"
    SUSPENDED = "suspended"
    FINISHED  = "finished"
    ERROR     = "error"
