"""CROF Audit/Observer service — append-only event ledger.

Every significant CROF operation emits an :class:`AuditEvent`.  Events are
appended to an :class:`AuditLedger` that provides:

* **Tamper evidence** via a rolling SHA-256 hash chain (each event records
  the hash of the previous event's canonical representation).
* **Streaming** — registered observers are called synchronously on each
  ``record()`` call; in production they would stream over gRPC
  ``AuditService.StreamAudit``.
* **Querying** — filter by source, event type, actor, and time window.

Event types (extensible)
------------------------
``envelope.dispatched``   ``envelope.validated``   ``envelope.rejected``
``identity.registered``   ``identity.revoked``
``transform.executed``    ``transform.failed``
``adaptation.triggered``  ``adaptation.safety_violation``
``interop.packet_sent``   ``interop.dvn_verified``
``policy.authorized``     ``policy.denied``
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single immutable audit record.

    Attributes
    ----------
    event_id:
        Unique identifier (UUIDv4).
    event_type:
        Semantic type string (dot-separated hierarchy).
    source:
        Identifier of the component that emitted the event.
    actor_id:
        Subject or service that caused the event.
    domain_id:
        Domain in which the event occurred.
    payload:
        Arbitrary structured data describing the event.
    timestamp:
        Unix epoch (seconds).
    prev_hash:
        SHA-256 hex digest of the previous event in the ledger (empty for
        the first event).
    event_hash:
        SHA-256 hex digest of this event's canonical representation.
        Computed automatically when appended to a ledger.
    """
    event_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source:     str = ""
    actor_id:   str = ""
    domain_id:  str = ""
    payload:    dict[str, Any] = field(default_factory=dict)
    timestamp:  int = field(default_factory=lambda: int(time.time()))
    prev_hash:  str = ""
    event_hash: str = ""

    def canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic representation (excludes ``event_hash``)."""
        return {
            "event_id":   self.event_id,
            "event_type": self.event_type,
            "source":     self.source,
            "actor_id":   self.actor_id,
            "domain_id":  self.domain_id,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
            "prev_hash":  self.prev_hash,
        }

    def compute_hash(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True,
                         separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def __repr__(self) -> str:
        return (
            f"AuditEvent({self.event_type!r}, actor={self.actor_id!r}, "
            f"ts={self.timestamp})"
        )


# ---------------------------------------------------------------------------
# AuditLedger  (Class-8 Audit/Observer Node)
# ---------------------------------------------------------------------------

class AuditLedger:
    """Append-only, hash-chained audit ledger.

    Parameters
    ----------
    name:
        Ledger identifier (e.g. domain name or service name).
    max_events:
        If set, discard the oldest events once the ledger exceeds this size
        (in production use a time-series database or log sink instead).
    """

    def __init__(self, name: str = "audit", max_events: int | None = None) -> None:
        self.name       = name
        self._max       = max_events
        self._events:   list[AuditEvent] = []
        self._observers: list[Callable[[AuditEvent], None]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: AuditEvent) -> AuditEvent:
        """Append *event* to the ledger.

        The ``prev_hash`` and ``event_hash`` fields are computed
        automatically.  All registered observers are notified.

        Returns
        -------
        AuditEvent
            The event as stored (with hashes populated).
        """
        event.prev_hash  = self._events[-1].event_hash if self._events else ""
        event.event_hash = event.compute_hash()

        self._events.append(event)

        if self._max is not None and len(self._events) > self._max:
            self._events.pop(0)

        for obs in self._observers:
            try:
                obs(event)
            except Exception:  # noqa: BLE001
                pass

        return event

    def emit(
        self,
        event_type: str,
        source: str = "",
        actor_id: str = "",
        domain_id: str = "",
        **payload: Any,
    ) -> AuditEvent:
        """Convenience builder — construct and record an event in one call."""
        ev = AuditEvent(
            event_type = event_type,
            source     = source,
            actor_id   = actor_id,
            domain_id  = domain_id,
            payload    = dict(payload),
        )
        return self.record(ev)

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def subscribe(self, observer: Callable[[AuditEvent], None]) -> None:
        """Register an observer that is called for every new event."""
        self._observers.append(observer)

    def unsubscribe(self, observer: Callable[[AuditEvent], None]) -> None:
        self._observers = [o for o in self._observers if o is not observer]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def all_events(self) -> list[AuditEvent]:
        return list(self._events)

    def filter(
        self,
        event_type:  str | None = None,
        source:      str | None = None,
        actor_id:    str | None = None,
        domain_id:   str | None = None,
        since:       int | None = None,
        until:       int | None = None,
    ) -> list[AuditEvent]:
        """Return events matching all provided non-``None`` criteria."""
        result = self._events
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if source is not None:
            result = [e for e in result if e.source == source]
        if actor_id is not None:
            result = [e for e in result if e.actor_id == actor_id]
        if domain_id is not None:
            result = [e for e in result if e.domain_id == domain_id]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        if until is not None:
            result = [e for e in result if e.timestamp <= until]
        return list(result)

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the hash chain integrity of the ledger.

        Returns
        -------
        (ok, issues)
            *ok* is ``True`` when no integrity problems are found.
            *issues* lists human-readable descriptions of any problems.
        """
        issues: list[str] = []
        prev   = ""
        for ev in self._events:
            if ev.prev_hash != prev:
                issues.append(
                    f"Chain break at {ev.event_id!r}: "
                    f"expected prev_hash={prev!r}, got {ev.prev_hash!r}"
                )
            expected = ev.compute_hash()
            if ev.event_hash != expected:
                issues.append(
                    f"Hash mismatch at {ev.event_id!r}: "
                    f"stored={ev.event_hash!r}, computed={expected!r}"
                )
            prev = ev.event_hash
        return (len(issues) == 0), issues

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._events)

    @property
    def latest(self) -> AuditEvent | None:
        return self._events[-1] if self._events else None

    def __repr__(self) -> str:
        return f"AuditLedger(name={self.name!r}, events={self.count})"
