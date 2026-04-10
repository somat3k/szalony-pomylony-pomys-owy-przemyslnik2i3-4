"""Tensor hyperparameter management.

Provides a typed dataclass for individual hyperparameters and a
:class:`HyperParamSpace` that holds a named collection of them — making it
easy to sweep, mutate, or optimise the full set of parameters that govern
tensor operations, model training, and shard-level behaviour.

Each :class:`HyperParameter` can be linked to a
:class:`~hololang.tensor.matrix.ParameterMultiplier` so that its effective
value changes over time according to a scheduling policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator

from hololang.tensor.matrix import ParameterMultiplier


# ---------------------------------------------------------------------------
# HyperParameter
# ---------------------------------------------------------------------------

@dataclass
class HyperParameter:
    """A single named hyperparameter with optional scheduling.

    Parameters
    ----------
    name:
        Unique identifier within a :class:`HyperParamSpace`.
    value:
        Current (or base) scalar value.
    dtype:
        Semantic type tag — ``"float"``, ``"int"``, ``"bool"``.
    description:
        Human-readable description.
    min_value:
        Minimum legal value (used for clamping / validation).
    max_value:
        Maximum legal value.
    multiplier:
        Optional :class:`~hololang.tensor.matrix.ParameterMultiplier` that
        modifies *value* dynamically when :meth:`effective_value` is called.
    """

    name:        str
    value:       float
    dtype:       str = "float"
    description: str = ""
    min_value:   float | None = None
    max_value:   float | None = None
    multiplier:  ParameterMultiplier | None = None

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def effective_value(self, step: int | None = None) -> float:
        """Return *value* scaled by the multiplier (if any).

        Parameters
        ----------
        step:
            Override the multiplier's internal step counter.
        """
        v = self.value
        if self.multiplier is not None:
            v *= self.multiplier.value(step)
        if self.min_value is not None:
            v = max(self.min_value, v)
        if self.max_value is not None:
            v = min(self.max_value, v)
        return v

    def step(self) -> float:
        """Advance the multiplier and return the new effective value."""
        if self.multiplier is not None:
            self.multiplier.step()
        return self.effective_value()

    def clamp(self) -> "HyperParameter":
        """Return a copy with *value* clamped to ``[min_value, max_value]``."""
        v = self.value
        if self.min_value is not None:
            v = max(self.min_value, v)
        if self.max_value is not None:
            v = min(self.max_value, v)
        p = copy.copy(self)
        p.value = v
        return p

    def __repr__(self) -> str:
        return (
            f"HyperParameter({self.name!r}={self.value}, "
            f"dtype={self.dtype!r}, "
            f"range=[{self.min_value}, {self.max_value}])"
        )


# ---------------------------------------------------------------------------
# HyperParamSpace
# ---------------------------------------------------------------------------

class HyperParamSpace:
    """Named collection of :class:`HyperParameter` instances.

    Provides dict-like access, bulk stepping, serialization, and
    random/grid search helpers.

    Parameters
    ----------
    name:
        Identifier for the parameter space.
    """

    def __init__(self, name: str = "params") -> None:
        self.name = name
        self._params: dict[str, HyperParameter] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        value: float,
        dtype: str = "float",
        description: str = "",
        min_value: float | None = None,
        max_value: float | None = None,
        multiplier: ParameterMultiplier | None = None,
    ) -> "HyperParamSpace":
        """Register a new hyperparameter.

        Returns *self* for fluent chaining.
        """
        self._params[name] = HyperParameter(
            name=name,
            value=value,
            dtype=dtype,
            description=description,
            min_value=min_value,
            max_value=max_value,
            multiplier=multiplier,
        )
        return self

    def register(self, param: HyperParameter) -> "HyperParamSpace":
        """Register a pre-built :class:`HyperParameter`."""
        self._params[param.name] = param
        return self

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str) -> HyperParameter:
        """Return the :class:`HyperParameter` for *name*."""
        if name not in self._params:
            raise KeyError(f"HyperParamSpace {self.name!r}: unknown param {name!r}")
        return self._params[name]

    def value(self, name: str, step: int | None = None) -> float:
        """Shortcut: return the effective value of param *name*."""
        return self.get(name).effective_value(step)

    def __getitem__(self, name: str) -> HyperParameter:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self) -> Iterator[HyperParameter]:
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def step_all(self) -> dict[str, float]:
        """Advance every multiplier and return a snapshot of effective values."""
        return {name: p.step() for name, p in self._params.items()}

    def effective_values(self, step: int | None = None) -> dict[str, float]:
        """Return a snapshot of all effective values at *step*."""
        return {name: p.effective_value(step) for name, p in self._params.items()}

    def reset_all(self) -> None:
        """Reset every multiplier's internal step counter."""
        for p in self._params.values():
            if p.multiplier is not None:
                p.multiplier.reset()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export the space as a plain dict (values only, no multipliers)."""
        return {
            name: {
                "value":       p.value,
                "dtype":       p.dtype,
                "description": p.description,
                "min_value":   p.min_value,
                "max_value":   p.max_value,
            }
            for name, p in self._params.items()
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], name: str = "params") -> "HyperParamSpace":
        """Reconstruct a :class:`HyperParamSpace` from a plain dict."""
        space = cls(name)
        for param_name, cfg in d.items():
            space.add(
                name=param_name,
                value=cfg.get("value", 0.0),
                dtype=cfg.get("dtype", "float"),
                description=cfg.get("description", ""),
                min_value=cfg.get("min_value"),
                max_value=cfg.get("max_value"),
            )
        return space

    def __repr__(self) -> str:
        return (
            f"HyperParamSpace(name={self.name!r}, "
            f"params={list(self._params)})"
        )
