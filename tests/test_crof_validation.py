"""Tests for CROF Validation Plane (Plane C) — 4-pass pipeline."""

import time
import pytest

from hololang.crof.envelope import Envelope, VisibilityClass, OperationType
from hololang.crof.identity import Subject, Capability, IdentityStore, SubjectKind
from hololang.crof.validation import Validator, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_with_actor(cap_names: list[str]) -> tuple[IdentityStore, Subject]:
    store = IdentityStore("test")
    s = Subject(kind=SubjectKind.SERVICE, name="svc")
    for cap in cap_names:
        s.grant_capability(Capability(cap))
    store.register(s)
    return store, s


def _signed_env(actor: Subject, **kw) -> Envelope:
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test", **kw)
    return env.sign(actor._secret)


# ---------------------------------------------------------------------------
# Pass 1 — Schema validation
# ---------------------------------------------------------------------------

def test_schema_empty_actor_id_fails():
    store, _ = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id="", domain_id="test")
    result = v.validate(env)
    assert not result.passed
    assert not result.schema_ok
    assert any("actor_id" in m for m in result.messages)


def test_schema_empty_domain_id_fails():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="")
    result = v.validate(env)
    assert not result.passed
    assert any("domain_id" in m for m in result.messages)


def test_schema_bad_visibility_fails():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="d",
                          visibility_class="SuperSecret")
    result = v.validate(env)
    assert not result.passed
    assert any("visibility_class" in m for m in result.messages)


def test_schema_expired_envelope_fails():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    now = int(time.time())
    env = Envelope(
        actor_id  = actor.subject_id,
        domain_id = "d",
        issued_at = now - 7200,
        expires_at= now - 1,
    )
    result = v.validate(env)
    assert not result.passed
    assert any("window" in m.lower() for m in result.messages)


# ---------------------------------------------------------------------------
# Pass 2 — Cryptographic validation
# ---------------------------------------------------------------------------

def test_crypto_valid_signature():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store)
    env = _signed_env(actor)
    result = v.validate(env)
    assert result.crypto_ok


def test_crypto_invalid_signature():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store)  # allow_unsigned=False
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test")
    env = env.sign(b"wrong-key")  # signed with wrong key
    result = v.validate(env)
    assert not result.passed
    assert not result.crypto_ok


def test_crypto_allow_unsigned():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test")
    # No signature at all
    result = v.validate(env)
    assert result.crypto_ok   # soft-passed


# ---------------------------------------------------------------------------
# Pass 3 — Policy validation
# ---------------------------------------------------------------------------

def test_policy_unknown_actor_fails():
    store = IdentityStore("test")
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id="unknown-actor", domain_id="test")
    result = v.validate(env)
    assert not result.passed
    assert any("unknown actor" in m.lower() for m in result.messages)


def test_policy_revoked_actor_fails():
    store, actor = _store_with_actor(["fabric.query"])
    store.revoke(actor.subject_id)
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test")
    result = v.validate(env)
    assert not result.passed
    assert any("revoked" in m.lower() for m in result.messages)


def test_policy_missing_capability_fails():
    store, actor = _store_with_actor([])  # no capabilities
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          operation_type=OperationType.COMMAND.value)
    result = v.validate(env)
    assert not result.passed
    assert any("capability" in m.lower() for m in result.messages)


def test_policy_correct_capability_passes():
    store, actor = _store_with_actor(["fabric.command"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          operation_type=OperationType.COMMAND.value)
    result = v.validate(env)
    assert result.policy_ok


# ---------------------------------------------------------------------------
# Pass 4 — Context validation
# ---------------------------------------------------------------------------

def test_context_sealed_without_cap_fails():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          visibility_class=VisibilityClass.SEALED.value)
    result = v.validate(env)
    assert not result.passed
    assert any("sealed" in m.lower() for m in result.messages)


def test_context_sealed_with_cap_passes():
    store, actor = _store_with_actor(["fabric.query", "context.sealed.access"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          visibility_class=VisibilityClass.SEALED.value)
    result = v.validate(env)
    assert result.context_ok


def test_context_restricted_without_cap_fails():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          visibility_class=VisibilityClass.RESTRICTED.value)
    result = v.validate(env)
    assert not result.context_ok


def test_context_open_always_passes():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store, allow_unsigned=True)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test",
                          visibility_class=VisibilityClass.OPEN.value)
    result = v.validate(env)
    assert result.context_ok


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_pass():
    store, actor = _store_with_actor(["fabric.query"])
    v = Validator(store)
    env = _signed_env(actor, visibility_class=VisibilityClass.OPEN.value)
    result = v.validate(env)
    assert result.passed
    assert result.schema_ok
    assert result.crypto_ok
    assert result.policy_ok
    assert result.context_ok


def test_custom_required_capabilities():
    store, actor = _store_with_actor(["custom.read"])
    v = Validator(
        store,
        allow_unsigned=True,
        required_capabilities={OperationType.QUERY.value: "custom.read"},
    )
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test")
    result = v.validate(env)
    assert result.policy_ok


def test_validation_result_repr():
    vr = ValidationResult(envelope_id="abc", passed=True,
                           schema_ok=True, crypto_ok=True,
                           policy_ok=True, context_ok=True)
    assert "PASS" in repr(vr)
