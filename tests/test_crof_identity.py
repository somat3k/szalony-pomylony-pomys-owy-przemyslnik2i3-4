"""Tests for CROF Identity Plane (Plane B)."""

import time
import pytest
from hololang.crof.identity import (
    Subject, Capability, RoleBinding, IdentityStore, SubjectKind,
)


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

def test_capability_active_no_expiry():
    cap = Capability("fabric.query")
    assert cap.is_active()


def test_capability_expired():
    cap = Capability("fabric.query", expires_at=int(time.time()) - 1)
    assert not cap.is_active()


def test_capability_not_yet_expired():
    cap = Capability("fabric.command", expires_at=int(time.time()) + 3600)
    assert cap.is_active()


# ---------------------------------------------------------------------------
# Subject — capabilities
# ---------------------------------------------------------------------------

def test_grant_and_has_capability():
    s = Subject(kind=SubjectKind.SERVICE)
    cap = Capability("fabric.transform")
    s.grant_capability(cap)
    assert s.has_capability("fabric.transform")


def test_missing_capability():
    s = Subject(kind=SubjectKind.SERVICE)
    assert not s.has_capability("fabric.admin")


def test_revoke_capability():
    s = Subject()
    s.grant_capability(Capability("fabric.event"))
    s.revoke_capability("fabric.event")
    assert not s.has_capability("fabric.event")


def test_active_capabilities_filters_expired():
    s = Subject()
    s.grant_capability(Capability("active", expires_at=int(time.time()) + 3600))
    s.grant_capability(Capability("expired", expires_at=int(time.time()) - 1))
    active = s.active_capabilities()
    assert len(active) == 1
    assert active[0].name == "active"


# ---------------------------------------------------------------------------
# Subject — role bindings
# ---------------------------------------------------------------------------

def test_bind_and_has_role():
    s = Subject()
    s.bind_role("admin", scope="domain:analytics")
    assert s.has_role("admin", scope="domain:analytics")


def test_role_wrong_scope():
    s = Subject()
    s.bind_role("operator", scope="domain:A")
    assert not s.has_role("operator", scope="domain:B")


def test_role_any_scope():
    s = Subject()
    s.bind_role("viewer")
    assert s.has_role("viewer")


# ---------------------------------------------------------------------------
# Subject — serialisation
# ---------------------------------------------------------------------------

def test_subject_to_dict():
    s = Subject(kind=SubjectKind.AGENT, name="my-agent")
    s.grant_capability(Capability("cap.x", scope="domain:A"))
    d = s.to_dict()
    assert d["kind"] == SubjectKind.AGENT
    assert d["name"] == "my-agent"
    assert any(c["name"] == "cap.x" for c in d["capabilities"])


def test_fingerprint_stable():
    s = Subject(subject_id="fixed-id", kind=SubjectKind.HUMAN, name="Alice")
    fp1 = s.fingerprint()
    fp2 = s.fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 16


# ---------------------------------------------------------------------------
# IdentityStore
# ---------------------------------------------------------------------------

def test_register_and_get():
    store = IdentityStore("test-domain")
    s = Subject(kind=SubjectKind.SERVICE, name="svc-1")
    store.register(s)
    assert store.get(s.subject_id) is s


def test_double_register_raises():
    store = IdentityStore()
    s = Subject()
    store.register(s)
    with pytest.raises(ValueError, match="already registered"):
        store.register(s)


def test_require_missing_raises():
    store = IdentityStore()
    with pytest.raises(KeyError):
        store.require("non-existent-id")


def test_revoke():
    store = IdentityStore()
    s = Subject()
    store.register(s)
    store.revoke(s.subject_id)
    assert store.get(s.subject_id).revoked


def test_remove():
    store = IdentityStore()
    s = Subject()
    store.register(s)
    store.remove(s.subject_id)
    assert store.get(s.subject_id) is None


def test_list_subjects_all():
    store = IdentityStore()
    store.register(Subject(kind=SubjectKind.HUMAN))
    store.register(Subject(kind=SubjectKind.SERVICE))
    assert len(store.list_subjects()) == 2


def test_list_subjects_by_kind():
    store = IdentityStore()
    store.register(Subject(kind=SubjectKind.HUMAN))
    store.register(Subject(kind=SubjectKind.SERVICE))
    humans = store.list_subjects(kind=SubjectKind.HUMAN)
    assert len(humans) == 1


def test_find_by_capability():
    store = IdentityStore()
    s1 = Subject()
    s1.grant_capability(Capability("special.cap"))
    s2 = Subject()
    store.register(s1)
    store.register(s2)
    found = store.find_by_capability("special.cap")
    assert len(found) == 1
    assert found[0].subject_id == s1.subject_id


def test_find_by_role():
    store = IdentityStore()
    s1 = Subject()
    s1.bind_role("ops")
    s2 = Subject()
    store.register(s1)
    store.register(s2)
    found = store.find_by_role("ops")
    assert len(found) == 1


def test_count():
    store = IdentityStore()
    assert store.count == 0
    store.register(Subject())
    assert store.count == 1
