"""Integration tests for the full CROFFabric orchestrator."""

import pytest
from hololang.crof.fabric import CROFFabric
from hololang.crof.envelope import Envelope, OperationType, VisibilityClass
from hololang.crof.identity import Subject, Capability, SubjectKind
from hololang.crof.interop import LayerZeroAdapter, GrpcAdapter
from hololang.crof.transform import TransformModule
from hololang.crof.adaptation import AdaptationRule, Telemetry
from hololang.crof.validation import ValidationResult
from hololang.crof.interop import DispatchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fabric(allow_unsigned: bool = True) -> CROFFabric:
    return CROFFabric("test-domain", allow_unsigned=allow_unsigned)


def _actor_with_caps(fabric: CROFFabric, *cap_names: str) -> Subject:
    s = Subject(kind=SubjectKind.SERVICE, name="svc")
    for cap in cap_names:
        s.grant_capability(Capability(cap))
    fabric.identity.register(s)
    return s


def _signed_env(actor: Subject, fabric: CROFFabric, **kw) -> Envelope:
    env = Envelope.build(
        actor_id=actor.subject_id,
        domain_id=fabric.domain_id,
        **kw,
    )
    return env.sign(actor._secret)


# ---------------------------------------------------------------------------
# Fabric construction
# ---------------------------------------------------------------------------

def test_fabric_repr():
    fabric = _fabric()
    r = repr(fabric)
    assert "test-domain" in r


def test_fabric_domain_id():
    fabric = CROFFabric("analytics")
    assert fabric.domain_id == "analytics"


# ---------------------------------------------------------------------------
# Identity registration
# ---------------------------------------------------------------------------

def test_register_actor():
    fabric = _fabric()
    s = _actor_with_caps(fabric, "fabric.query")
    assert fabric.identity.get(s.subject_id) is s


# ---------------------------------------------------------------------------
# Dispatch — validation pass/fail
# ---------------------------------------------------------------------------

def test_dispatch_valid_envelope():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    env = _signed_env(actor, fabric)
    result = fabric.dispatch(env)
    assert isinstance(result, DispatchResult)
    assert result.ok


def test_dispatch_invalid_envelope_returns_validation_result():
    fabric = _fabric(allow_unsigned=False)
    actor = _actor_with_caps(fabric, "fabric.query")
    # Envelope with empty domain_id — schema fail
    env = Envelope.build(actor_id=actor.subject_id, domain_id="")
    result = fabric.dispatch(env)
    assert isinstance(result, ValidationResult)
    assert not result.passed


def test_dispatch_without_validation():
    fabric = _fabric()
    actor = _actor_with_caps(fabric)
    env = Envelope.build(actor_id=actor.subject_id, domain_id="test-domain")
    result = fabric.dispatch(env, validate=False)
    assert isinstance(result, DispatchResult)


# ---------------------------------------------------------------------------
# Dispatch — adapter routing
# ---------------------------------------------------------------------------

def test_dispatch_to_layerzero_adapter():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    lz = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    fabric.register_adapter("lz", lz)
    env = _signed_env(actor, fabric)
    result = fabric.dispatch(env, adapter_name="lz")
    assert isinstance(result, DispatchResult)
    assert result.adapter_type == "layerzero"
    assert len(lz.sent_packets) == 1


def test_dispatch_to_grpc_adapter():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    grpc = GrpcAdapter(endpoint="localhost:50051")
    fabric.register_adapter("grpc", grpc)
    env = _signed_env(actor, fabric)
    result = fabric.dispatch(env, adapter_name="grpc")
    assert result.adapter_type == "grpc"


def test_dispatch_unknown_adapter_raises():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    env = _signed_env(actor, fabric)
    with pytest.raises(KeyError):
        fabric.dispatch(env, adapter_name="missing-adapter")


# ---------------------------------------------------------------------------
# Transform graph integration
# ---------------------------------------------------------------------------

def test_dispatch_runs_transform_graph():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")

    results: list[str] = []

    mod = TransformModule(
        module_id="echo",
        fn=lambda env, ctx: results.append("called") or b"ok",
    )
    fabric.transforms.add(mod)

    env = _signed_env(actor, fabric)
    result = fabric.dispatch(env)
    assert isinstance(result, DispatchResult)
    assert "called" in results


# ---------------------------------------------------------------------------
# Authorize
# ---------------------------------------------------------------------------

def test_authorize_with_capability():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "admin.manage")
    assert fabric.authorize(actor.subject_id, "admin.manage")


def test_authorize_without_capability():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    assert not fabric.authorize(actor.subject_id, "admin.manage")


def test_authorize_unknown_actor():
    fabric = _fabric()
    assert not fabric.authorize("ghost", "fabric.query")


def test_authorize_revoked_actor():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    fabric.identity.revoke(actor.subject_id)
    assert not fabric.authorize(actor.subject_id, "fabric.query")


# ---------------------------------------------------------------------------
# Adaptation loop integration
# ---------------------------------------------------------------------------

def test_tick_returns_summary():
    fabric = _fabric()
    summary = fabric.tick()
    assert "cycle" in summary


def test_tick_increments_audit():
    fabric = _fabric()
    initial = fabric.audit.count
    fabric.tick()
    assert fabric.audit.count > initial


def test_register_telemetry_source_and_rule():
    fabric = _fabric()
    fired: list[bool] = []

    fabric.register_telemetry_source(
        lambda: Telemetry(source="svc", metrics={"value": 100.0})
    )
    fabric.register_rule(AdaptationRule(
        rule_id="always-fire",
        condition=lambda state: True,
        action=lambda state: fired.append(True) or "ok",
    ))
    fabric.tick()
    assert len(fired) == 1


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_audit_records_validation_events():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    env = _signed_env(actor, fabric)
    fabric.dispatch(env)
    events = fabric.audit.filter(event_type="envelope.validated")
    assert len(events) >= 1


def test_audit_records_rejection():
    fabric = _fabric(allow_unsigned=False)
    # Actor with no signing secret registered — signature will fail
    env = Envelope.build(actor_id="fake-actor", domain_id="test-domain")
    fabric.dispatch(env)
    events = fabric.audit.filter(event_type="envelope.rejected")
    assert len(events) >= 1


def test_audit_chain_integrity():
    fabric = _fabric()
    actor = _actor_with_caps(fabric, "fabric.query")
    for _ in range(5):
        env = _signed_env(actor, fabric)
        fabric.dispatch(env)
    ok, issues = fabric.audit.verify_chain()
    assert ok
    assert issues == []
