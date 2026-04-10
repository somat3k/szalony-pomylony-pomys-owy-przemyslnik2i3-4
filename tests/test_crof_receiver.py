"""Tests for Receiver aspects and ReceiverChain (hololang/crof/receiver.py)."""

import pytest

from hololang.crof.envelope import Envelope, OperationType
from hololang.crof.receiver import Receiver, ReceiverChain, ReceiverDecision


def _env(**kw) -> Envelope:
    return Envelope.build(actor_id="test-actor", domain_id="test-domain", **kw)


# ===========================================================================
# ReceiverDecision helpers
# ===========================================================================

def test_decision_routes_to_model():
    d = ReceiverDecision(matched=True, target_layer="model", model_name="m")
    assert d.routes_to_model


def test_decision_routes_to_embedding():
    d = ReceiverDecision(matched=True, target_layer="embedding")
    assert d.routes_to_embedding


def test_decision_routes_to_tensor():
    d = ReceiverDecision(matched=True, target_layer="tensor")
    assert d.routes_to_tensor


def test_decision_passthrough_no_route():
    d = ReceiverDecision(matched=False, target_layer="passthrough")
    assert not d.routes_to_model
    assert not d.routes_to_embedding
    assert not d.routes_to_tensor


def test_decision_repr_matched():
    d = ReceiverDecision(matched=True, receiver_name="r1", target_layer="model", model_name="m")
    assert "r1" in repr(d)
    assert "model" in repr(d)


def test_decision_repr_unmatched():
    d = ReceiverDecision(matched=False)
    assert "False" in repr(d)


# ===========================================================================
# Receiver — basic matching
# ===========================================================================

def test_receiver_matches_always_true():
    r = Receiver(predicate=lambda env: True)
    assert r.matches(_env())


def test_receiver_matches_always_false():
    r = Receiver(predicate=lambda env: False)
    assert not r.matches(_env())


def test_receiver_predicate_uses_envelope():
    r = Receiver(
        predicate=lambda env: env.operation_type == OperationType.TRANSFORM.value
    )
    assert r.matches(_env(operation_type=OperationType.TRANSFORM.value))
    assert not r.matches(_env(operation_type=OperationType.QUERY.value))


def test_receiver_disabled_never_matches():
    r = Receiver(predicate=lambda env: True, enabled=False)
    assert not r.matches(_env())


def test_receiver_predicate_exception_treated_as_miss():
    def bad_pred(env):
        raise RuntimeError("oops")

    r = Receiver(predicate=bad_pred)
    assert not r.matches(_env())


def test_receiver_claim_count_increments():
    r = Receiver(predicate=lambda env: True)
    r.matches(_env())
    r.matches(_env())
    assert r.claim_count == 2


def test_receiver_miss_count_increments():
    r = Receiver(predicate=lambda env: False)
    r.matches(_env())
    assert r.miss_count == 1


def test_receiver_decide_produces_decision():
    r = Receiver(
        name="my-recv",
        target_layer="model",
        model_name="scorer",
        batch_size=64,
        priority=5,
    )
    d = r.decide(_env())
    assert d.matched
    assert d.receiver_name == "my-recv"
    assert d.target_layer  == "model"
    assert d.model_name    == "scorer"
    assert d.batch_size    == 64
    assert d.priority      == 5


def test_receiver_repr():
    r = Receiver(name="r1", enabled=True)
    assert "r1" in repr(r)
    assert "enabled" in repr(r)


def test_receiver_default_name_is_not_empty():
    r = Receiver()
    assert r.name != ""


# ===========================================================================
# ReceiverChain — basic
# ===========================================================================

def test_chain_empty_no_match():
    chain = ReceiverChain("c")
    d = chain.resolve(_env())
    assert not d.matched


def test_chain_single_receiver_match():
    chain = ReceiverChain()
    chain.add(Receiver(name="r1", predicate=lambda env: True, target_layer="tensor"))
    d = chain.resolve(_env())
    assert d.matched
    assert d.receiver_name == "r1"


def test_chain_single_receiver_no_match():
    chain = ReceiverChain()
    chain.add(Receiver(name="r1", predicate=lambda env: False))
    d = chain.resolve(_env())
    assert not d.matched


def test_chain_first_match_wins():
    chain = ReceiverChain()
    chain.add(Receiver(name="high", predicate=lambda e: True, priority=10, target_layer="model", model_name="A"))
    chain.add(Receiver(name="low",  predicate=lambda e: True, priority=1,  target_layer="tensor"))
    d = chain.resolve(_env())
    assert d.receiver_name == "high"
    assert d.model_name == "A"


def test_chain_priority_ordering():
    chain = ReceiverChain()
    # Add in reverse priority order
    chain.add(Receiver(name="low",  predicate=lambda e: True, priority=1, target_layer="tensor"))
    chain.add(Receiver(name="high", predicate=lambda e: True, priority=9, target_layer="model", model_name="M"))
    d = chain.resolve(_env())
    assert d.receiver_name == "high"


def test_chain_fallback_to_second_receiver():
    chain = ReceiverChain()
    chain.add(Receiver(name="first",  predicate=lambda e: False, priority=10))
    chain.add(Receiver(name="second", predicate=lambda e: True,  priority=1, target_layer="tensor"))
    d = chain.resolve(_env())
    assert d.receiver_name == "second"


def test_chain_resolve_count():
    chain = ReceiverChain()
    chain.add(Receiver(predicate=lambda e: True))
    chain.resolve(_env())
    chain.resolve(_env())
    assert chain.resolve_count == 2


def test_chain_match_rate_all_match():
    chain = ReceiverChain()
    chain.add(Receiver(predicate=lambda e: True))
    chain.resolve(_env())
    chain.resolve(_env())
    assert chain.match_rate == 1.0


def test_chain_match_rate_no_match():
    chain = ReceiverChain()
    chain.add(Receiver(predicate=lambda e: False))
    chain.resolve(_env())
    assert chain.match_rate == 0.0


def test_chain_match_rate_zero_resolves():
    chain = ReceiverChain()
    assert chain.match_rate == 0.0


def test_chain_count():
    chain = ReceiverChain()
    chain.add(Receiver(name="a"))
    chain.add(Receiver(name="b"))
    assert chain.count == 2


def test_chain_remove():
    chain = ReceiverChain()
    chain.add(Receiver(name="del-me"))
    removed = chain.remove("del-me")
    assert removed
    assert chain.count == 0


def test_chain_remove_nonexistent():
    chain = ReceiverChain()
    assert not chain.remove("x")


def test_chain_enable_disable():
    chain = ReceiverChain()
    chain.add(Receiver(name="r", predicate=lambda e: True, target_layer="tensor"))
    chain.disable("r")
    assert not chain.resolve(_env()).matched
    chain.enable("r")
    assert chain.resolve(_env()).matched


def test_chain_receivers_snapshot():
    chain = ReceiverChain()
    chain.add(Receiver(name="a", priority=5))
    chain.add(Receiver(name="b", priority=10))
    snap = chain.receivers
    # Should be sorted by descending priority
    assert snap[0].name == "b"
    assert snap[1].name == "a"


# ===========================================================================
# ReceiverChain — resolve_all (fan-out)
# ===========================================================================

def test_chain_resolve_all_multiple_matches():
    chain = ReceiverChain()
    chain.add(Receiver(name="a", predicate=lambda e: True, priority=5))
    chain.add(Receiver(name="b", predicate=lambda e: True, priority=3))
    decisions = chain.resolve_all(_env())
    assert len(decisions) == 2
    names = [d.receiver_name for d in decisions]
    assert "a" in names
    assert "b" in names


def test_chain_resolve_all_no_match():
    chain = ReceiverChain()
    chain.add(Receiver(name="a", predicate=lambda e: False))
    decisions = chain.resolve_all(_env())
    assert decisions == []


# ===========================================================================
# ReceiverChain — stats
# ===========================================================================

def test_chain_stats():
    chain = ReceiverChain(name="stats-chain")
    chain.add(Receiver(name="r1", predicate=lambda e: True))
    chain.resolve(_env())
    chain.resolve(_env())
    s = chain.stats()
    assert s["name"] == "stats-chain"
    assert s["resolve_count"] == 2
    assert s["match_count"] == 2
    assert len(s["receiver_stats"]) == 1


def test_chain_repr():
    chain = ReceiverChain(name="test-chain")
    assert "test-chain" in repr(chain)


# ===========================================================================
# Practical scenario — payload-type routing
# ===========================================================================

def test_scenario_payload_type_routing():
    chain = ReceiverChain("price-routing")

    chain.add(Receiver(
        name="json-model",
        predicate=lambda env: env.payload_type == "application/json",
        target_layer="model",
        model_name="price-scorer",
        batch_size=64,
        priority=10,
    ))
    chain.add(Receiver(
        name="generic-tensor",
        predicate=lambda env: env.operation_type == OperationType.TRANSFORM.value,
        target_layer="tensor",
        batch_size=32,
        priority=5,
    ))

    json_env = _env(payload_type="application/json")
    d1 = chain.resolve(json_env)
    assert d1.routes_to_model
    assert d1.model_name == "price-scorer"
    assert d1.batch_size == 64

    transform_env = _env(operation_type=OperationType.TRANSFORM.value)
    d2 = chain.resolve(transform_env)
    assert d2.routes_to_tensor
    assert d2.batch_size == 32

    plain_env = _env()
    d3 = chain.resolve(plain_env)
    assert not d3.matched
