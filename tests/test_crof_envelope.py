"""Tests for the CROF Universal Envelope (Plane A)."""

import time
import pytest
from hololang.crof.envelope import (
    Envelope, VisibilityClass, OperationType,
)


def _env(**kw) -> Envelope:
    return Envelope.build(
        actor_id="actor-1",
        domain_id="test-domain",
        **kw,
    )


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

def test_envelope_defaults():
    env = _env()
    assert env.envelope_id
    assert env.actor_id == "actor-1"
    assert env.domain_id == "test-domain"
    assert env.visibility_class == VisibilityClass.OPEN.value
    assert env.operation_type == OperationType.QUERY.value


def test_envelope_build_with_payload():
    env = _env(payload=b"hello", operation_type=OperationType.COMMAND.value)
    assert env.payload == b"hello"
    assert env.operation_type == OperationType.COMMAND.value


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_from_dict():
    env = _env(payload=b"\x01\x02", visibility_class=VisibilityClass.SHARED.value)
    d = env.to_dict()
    env2 = Envelope.from_dict(d)
    assert env2.envelope_id == env.envelope_id
    assert env2.payload == env.payload
    assert env2.visibility_class == VisibilityClass.SHARED.value


def test_to_from_json():
    env = _env(subject_id="subject-X", evidence_refs=["ref1", "ref2"])
    j = env.to_json()
    env2 = Envelope.from_json(j)
    assert env2.subject_id == "subject-X"
    assert env2.evidence_refs == ["ref1", "ref2"]


def test_roundtrip_preserves_all_fields():
    env = _env(
        tenant_id       = "tenant-A",
        module_id       = "mod-1",
        capability_id   = "cap-1",
        policy_id       = "pol-1",
        context_id      = "ctx-1",
        visibility_class= VisibilityClass.RESTRICTED.value,
        operation_type  = OperationType.TRANSFORM.value,
        payload_type    = "application/json",
        payload         = b'{"key":"val"}',
        evidence_refs   = ["ev-1"],
        provenance_refs = ["prov-1"],
    )
    env2 = Envelope.from_dict(env.to_dict())
    assert env2.tenant_id        == "tenant-A"
    assert env2.module_id        == "mod-1"
    assert env2.capability_id    == "cap-1"
    assert env2.policy_id        == "pol-1"
    assert env2.context_id       == "ctx-1"
    assert env2.visibility_class == VisibilityClass.RESTRICTED.value
    assert env2.operation_type   == OperationType.TRANSFORM.value
    assert env2.payload          == b'{"key":"val"}'
    assert env2.evidence_refs    == ["ev-1"]
    assert env2.provenance_refs  == ["prov-1"]


# ---------------------------------------------------------------------------
# Signing & verification
# ---------------------------------------------------------------------------

def test_sign_and_verify():
    env    = _env(payload=b"data")
    key    = b"super-secret"
    signed = env.sign(key)
    assert signed.signature
    assert signed.verify_signature(key)


def test_wrong_key_fails_verification():
    env    = _env()
    signed = env.sign(b"key-A")
    assert not signed.verify_signature(b"key-B")


def test_tampered_payload_fails_verification():
    env    = _env(payload=b"original")
    key    = b"secret"
    signed = env.sign(key)
    # Tamper with the payload after signing
    import dataclasses
    tampered = dataclasses.replace(signed, payload=b"modified")
    assert not tampered.verify_signature(key)


def test_unsigned_envelope_fails_verification():
    env = _env()
    assert not env.verify_signature(b"any-key")


# ---------------------------------------------------------------------------
# Temporal validity
# ---------------------------------------------------------------------------

def test_is_expired_future():
    env = _env(ttl_seconds=3600)
    assert not env.is_expired()


def test_is_expired_past():
    now = int(time.time())
    env = Envelope(
        actor_id  = "a",
        domain_id = "d",
        issued_at = now - 7200,
        expires_at= now - 3600,
    )
    assert env.is_expired()


def test_is_valid_window():
    env = _env(ttl_seconds=3600)
    assert env.is_valid_window()


def test_invalid_window_expired():
    now = int(time.time())
    env = Envelope(
        actor_id  = "a",
        domain_id = "d",
        issued_at = now - 7200,
        expires_at= now - 1,
    )
    assert not env.is_valid_window()


# ---------------------------------------------------------------------------
# Visibility & operation enums
# ---------------------------------------------------------------------------

def test_all_visibility_classes():
    for vc in VisibilityClass:
        env = _env(visibility_class=vc.value)
        assert env.visibility_class == vc.value


def test_all_operation_types():
    for op in OperationType:
        env = _env(operation_type=op.value)
        assert env.operation_type == op.value
