"""Receiver aspects — envelope routing to tensor compute layers.

A :class:`Receiver` is a typed aspect that inspects an incoming
:class:`~hololang.crof.envelope.Envelope` and decides whether to claim it
for processing by a specific tensor compute layer or model deployment.

The pattern mirrors Aspect-Oriented Programming (AOP):

* **Pointcut** — the predicate that identifies matching envelopes.
* **Advice** — the routing decision (which model, which layer, which batch size).
* **Weaving** — done by :class:`ReceiverChain` which applies receivers in
  priority order and returns the first match.

This makes the CROF message layer *compute-aware* without coupling the
transport plane to any specific ML framework.

Receiver taxonomy
-----------------
Each receiver carries a ``target_layer`` string that tells the
:class:`~hololang.crof.compute.TensorComputeNode` which backend to use:

* ``"tensor"``    — raw tensor operation (no model)
* ``"model"``     — forward pass through a named model deployment
* ``"embedding"`` — embedding lookup / composition via GAS space
* ``"passthrough"`` — forward without compute (audit / logging only)

Usage::

    from hololang.crof.receiver import Receiver, ReceiverChain, ReceiverDecision

    chain = ReceiverChain("inference-chain")

    chain.add(Receiver(
        name="price-feed-receiver",
        predicate=lambda env: env.payload_type == "application/json"
                              and env.module_id == "price-oracle",
        target_layer="model",
        model_name="arbitrage-scorer",
        batch_size=64,
        priority=10,
    ))

    chain.add(Receiver(
        name="embedding-receiver",
        predicate=lambda env: env.operation_type == "Transform",
        target_layer="embedding",
        priority=5,
    ))

    decision = chain.resolve(envelope)
    if decision.matched:
        node.route(envelope, decision)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from hololang.crof.envelope import Envelope


# ---------------------------------------------------------------------------
# ReceiverDecision — result of chain resolution
# ---------------------------------------------------------------------------

@dataclass
class ReceiverDecision:
    """Result returned by :meth:`ReceiverChain.resolve`.

    Attributes
    ----------
    matched:
        ``True`` when a receiver claimed the envelope.
    receiver_name:
        Name of the receiver that claimed it (empty when *matched* is ``False``).
    target_layer:
        The compute layer to invoke (``"tensor"``, ``"model"``, …).
    model_name:
        Name of the model deployment to invoke (may be empty).
    batch_size:
        Recommended batch size for spread-tensor partitioning.
    priority:
        Priority of the matched receiver.
    metadata:
        Receiver-specific extension fields.
    resolved_at:
        Unix timestamp when the decision was made.
    """
    matched:       bool           = False
    receiver_name: str            = ""
    target_layer:  str            = "passthrough"
    model_name:    str            = ""
    batch_size:    int            = 32
    priority:      int            = 0
    metadata:      dict[str, Any] = field(default_factory=dict)
    resolved_at:   float          = field(default_factory=time.time)

    @property
    def routes_to_model(self) -> bool:
        return self.matched and self.target_layer == "model"

    @property
    def routes_to_embedding(self) -> bool:
        return self.matched and self.target_layer == "embedding"

    @property
    def routes_to_tensor(self) -> bool:
        return self.matched and self.target_layer == "tensor"

    def __repr__(self) -> str:
        if not self.matched:
            return "ReceiverDecision(matched=False)"
        return (
            f"ReceiverDecision(receiver={self.receiver_name!r}, "
            f"layer={self.target_layer!r}, model={self.model_name!r})"
        )


# ---------------------------------------------------------------------------
# Receiver — a single named aspect
# ---------------------------------------------------------------------------

class Receiver:
    """A named aspect that claims matching envelopes for a compute target.

    Parameters
    ----------
    name:
        Unique name for this receiver (used for logging and debugging).
    predicate:
        Callable ``(Envelope) -> bool`` that returns ``True`` when this
        receiver should claim the envelope.
    target_layer:
        The compute layer the receiver routes to:
        ``"tensor"`` | ``"model"`` | ``"embedding"`` | ``"passthrough"``.
    model_name:
        Name of the model deployment (required when *target_layer* is
        ``"model"``; optional otherwise).
    batch_size:
        Recommended batch size when spreading tensors for this receiver.
    priority:
        Higher values are checked first in the :class:`ReceiverChain`.
        Receivers with equal priority are checked in insertion order.
    enabled:
        When ``False`` the receiver is skipped by the chain.
    metadata:
        Arbitrary key-value extension fields attached to every
        :class:`ReceiverDecision` produced by this receiver.
    """

    def __init__(
        self,
        name: str = "",
        predicate: Callable[[Envelope], bool] | None = None,
        target_layer: str = "passthrough",
        model_name: str = "",
        batch_size: int = 32,
        priority: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        import uuid as _uuid
        self.name         = name or str(_uuid.uuid4())[:8]
        self.predicate    = predicate or (lambda _env: True)
        self.target_layer = target_layer
        self.model_name   = model_name
        self.batch_size   = batch_size
        self.priority     = priority
        self.enabled      = enabled
        self.metadata     = dict(metadata or {})
        self._claims:     int = 0   # number of times this receiver claimed an envelope
        self._misses:     int = 0

    def matches(self, envelope: Envelope) -> bool:
        """Return ``True`` when the receiver's predicate fires on *envelope*.

        Returns ``False`` when the receiver is disabled or the predicate
        raises an exception (fail-safe: treat as no-match).
        """
        if not self.enabled:
            return False
        try:
            result = bool(self.predicate(envelope))
        except Exception:  # noqa: BLE001
            result = False
        if result:
            self._claims += 1
        else:
            self._misses += 1
        return result

    def decide(self, envelope: Envelope) -> ReceiverDecision:
        """Return a :class:`ReceiverDecision` for *envelope* (assuming match)."""
        return ReceiverDecision(
            matched       = True,
            receiver_name = self.name,
            target_layer  = self.target_layer,
            model_name    = self.model_name,
            batch_size    = self.batch_size,
            priority      = self.priority,
            metadata      = dict(self.metadata),
        )

    @property
    def claim_count(self) -> int:
        return self._claims

    @property
    def miss_count(self) -> int:
        return self._misses

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return (
            f"Receiver(name={self.name!r}, layer={self.target_layer!r}, "
            f"priority={self.priority}, {status}, claims={self._claims})"
        )


# ---------------------------------------------------------------------------
# ReceiverChain
# ---------------------------------------------------------------------------

class ReceiverChain:
    """Priority-ordered chain of :class:`Receiver` aspects.

    The chain evaluates receivers from highest priority to lowest and
    returns the first :class:`ReceiverDecision` with ``matched=True``.
    When no receiver matches, a default ``passthrough`` decision is returned
    (``matched=False``).

    Parameters
    ----------
    name:
        Identifier for this chain (used in logging).
    default_batch_size:
        Batch size placed in the passthrough decision when no receiver fires.
    """

    def __init__(
        self,
        name: str = "receiver-chain",
        default_batch_size: int = 32,
    ) -> None:
        self.name               = name
        self.default_batch_size = default_batch_size
        self._receivers: list[Receiver] = []
        self._resolve_count: int = 0
        self._match_count:   int = 0

    # ------------------------------------------------------------------
    # Receiver management
    # ------------------------------------------------------------------

    def add(self, receiver: Receiver) -> None:
        """Add a receiver and re-sort the chain by descending priority."""
        self._receivers.append(receiver)
        self._receivers.sort(key=lambda r: r.priority, reverse=True)

    def remove(self, name: str) -> bool:
        """Remove the first receiver with the given *name*.

        Returns ``True`` when a receiver was removed.
        """
        for i, r in enumerate(self._receivers):
            if r.name == name:
                del self._receivers[i]
                return True
        return False

    def enable(self, name: str) -> None:
        """Enable the named receiver."""
        for r in self._receivers:
            if r.name == name:
                r.enabled = True
                return

    def disable(self, name: str) -> None:
        """Disable the named receiver (it will be skipped by :meth:`resolve`)."""
        for r in self._receivers:
            if r.name == name:
                r.enabled = False
                return

    @property
    def receivers(self) -> list[Receiver]:
        """Ordered snapshot of active receivers (highest priority first)."""
        return list(self._receivers)

    @property
    def count(self) -> int:
        return len(self._receivers)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, envelope: Envelope) -> ReceiverDecision:
        """Find the first receiver that matches *envelope*.

        Receivers are evaluated in descending priority order.  The first
        receiver whose :meth:`~Receiver.matches` returns ``True`` wins.

        Returns
        -------
        ReceiverDecision
            ``matched=True`` when a receiver claimed the envelope;
            ``matched=False`` with ``target_layer="passthrough"`` otherwise.
        """
        self._resolve_count += 1
        for receiver in self._receivers:
            if receiver.matches(envelope):
                self._match_count += 1
                return receiver.decide(envelope)
        return ReceiverDecision(
            matched    = False,
            batch_size = self.default_batch_size,
        )

    def resolve_all(self, envelope: Envelope) -> list[ReceiverDecision]:
        """Return decisions from *all* matching receivers (for fan-out routing).

        Unlike :meth:`resolve`, this does not stop at the first match.
        """
        decisions: list[ReceiverDecision] = []
        for receiver in self._receivers:
            if receiver.matches(envelope):
                decisions.append(receiver.decide(envelope))
        return decisions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def resolve_count(self) -> int:
        """Total number of envelopes the chain has evaluated."""
        return self._resolve_count

    @property
    def match_rate(self) -> float:
        """Fraction of resolved envelopes that found a match [0, 1]."""
        if self._resolve_count == 0:
            return 0.0
        return self._match_count / self._resolve_count

    def stats(self) -> dict[str, Any]:
        """Return a summary dict of chain statistics."""
        return {
            "name":          self.name,
            "receivers":     self.count,
            "resolve_count": self._resolve_count,
            "match_count":   self._match_count,
            "match_rate":    self.match_rate,
            "receiver_stats": [
                {
                    "name":   r.name,
                    "claims": r.claim_count,
                    "misses": r.miss_count,
                    "enabled": r.enabled,
                }
                for r in self._receivers
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ReceiverChain(name={self.name!r}, receivers={self.count}, "
            f"resolved={self._resolve_count})"
        )
