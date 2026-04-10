"""Tests for CROF Audit/Observer service."""

import pytest
from hololang.crof.audit import AuditLedger, AuditEvent


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

def test_event_compute_hash_is_deterministic():
    ev = AuditEvent(
        event_id   = "fixed-id",
        event_type = "test.event",
        source     = "svc",
        actor_id   = "actor-1",
        domain_id  = "domain-A",
        payload    = {"key": "val"},
        timestamp  = 1_700_000_000,
        prev_hash  = "abc123",
    )
    h1 = ev.compute_hash()
    h2 = ev.compute_hash()
    assert h1 == h2
    assert len(h1) == 64   # SHA-256 hex


def test_event_repr():
    ev = AuditEvent(event_type="identity.registered", actor_id="a")
    assert "identity.registered" in repr(ev)


# ---------------------------------------------------------------------------
# AuditLedger — basic record / emit
# ---------------------------------------------------------------------------

def test_record_returns_event():
    ledger = AuditLedger()
    ev = AuditEvent(event_type="test.event", source="svc")
    result = ledger.record(ev)
    assert result is ev
    assert ev.event_hash  # should be populated


def test_emit_convenience():
    ledger = AuditLedger()
    ev = ledger.emit("transform.executed", source="transform-svc",
                     actor_id="actor-1", domain_id="d1", module_id="m1")
    assert ev.event_type == "transform.executed"
    assert ev.payload["module_id"] == "m1"


def test_count_grows():
    ledger = AuditLedger()
    assert ledger.count == 0
    ledger.emit("e1", source="s")
    ledger.emit("e2", source="s")
    assert ledger.count == 2


def test_latest():
    ledger = AuditLedger()
    assert ledger.latest is None
    ledger.emit("first", source="s")
    ev2 = ledger.emit("second", source="s")
    assert ledger.latest.event_type == "second"


# ---------------------------------------------------------------------------
# Hash chain
# ---------------------------------------------------------------------------

def test_hash_chain_linked():
    ledger = AuditLedger()
    ev1 = ledger.emit("e1", source="s")
    ev2 = ledger.emit("e2", source="s")
    assert ev1.prev_hash == ""          # first event has no predecessor
    assert ev2.prev_hash == ev1.event_hash


def test_verify_chain_intact():
    ledger = AuditLedger()
    for i in range(5):
        ledger.emit(f"event.{i}", source="s")
    ok, issues = ledger.verify_chain()
    assert ok
    assert issues == []


def test_verify_chain_tampered():
    ledger = AuditLedger()
    ev1 = ledger.emit("e1", source="s")
    ledger.emit("e2", source="s")
    # Tamper with first event's hash
    ev1.event_hash = "tampered"
    ok, issues = ledger.verify_chain()
    assert not ok
    assert len(issues) > 0


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

def test_observer_called_on_record():
    received: list[AuditEvent] = []
    ledger = AuditLedger()
    ledger.subscribe(received.append)
    ev = ledger.emit("test.obs", source="s")
    assert len(received) == 1
    assert received[0].event_id == ev.event_id


def test_multiple_observers():
    log1: list[str] = []
    log2: list[str] = []
    ledger = AuditLedger()
    ledger.subscribe(lambda e: log1.append(e.event_type))
    ledger.subscribe(lambda e: log2.append(e.event_type))
    ledger.emit("ping", source="s")
    assert len(log1) == 1
    assert len(log2) == 1


def test_unsubscribe():
    received: list[AuditEvent] = []

    def handler(ev: AuditEvent) -> None:
        received.append(ev)

    ledger = AuditLedger()
    ledger.subscribe(handler)
    ledger.unsubscribe(handler)
    ledger.emit("test", source="s")
    assert len(received) == 0


def test_observer_exception_silently_handled():
    def bad_obs(ev):
        raise RuntimeError("observer failure")

    ledger = AuditLedger()
    ledger.subscribe(bad_obs)
    ledger.emit("test", source="s")   # should not raise
    assert ledger.count == 1


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def test_all_events():
    ledger = AuditLedger()
    ledger.emit("a", source="s1")
    ledger.emit("b", source="s2")
    events = ledger.all_events()
    assert len(events) == 2


def test_filter_by_event_type():
    ledger = AuditLedger()
    ledger.emit("login", source="s")
    ledger.emit("logout", source="s")
    ledger.emit("login", source="s")
    logins = ledger.filter(event_type="login")
    assert len(logins) == 2


def test_filter_by_actor():
    ledger = AuditLedger()
    ledger.emit("e", source="s", actor_id="alice")
    ledger.emit("e", source="s", actor_id="bob")
    alices = ledger.filter(actor_id="alice")
    assert len(alices) == 1


def test_filter_by_since():
    import time
    ledger = AuditLedger()
    ev1 = AuditEvent(event_type="old", timestamp=1000)
    ev2 = AuditEvent(event_type="new", timestamp=2000)
    ledger.record(ev1)
    ledger.record(ev2)
    recent = ledger.filter(since=1500)
    assert len(recent) == 1
    assert recent[0].event_type == "new"


# ---------------------------------------------------------------------------
# max_events cap
# ---------------------------------------------------------------------------

def test_max_events_cap():
    ledger = AuditLedger(max_events=3)
    for i in range(5):
        ledger.emit(f"event.{i}", source="s")
    assert ledger.count == 3
