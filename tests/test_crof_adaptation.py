"""Tests for CROF Adaptation Plane (Plane G) — cybernetic loop."""

import time
import pytest
from hololang.crof.adaptation import (
    Telemetry, AdaptationRule, CyberneticLoop, RuleOutcome,
)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_telemetry_get_existing():
    t = Telemetry(source="svc", metrics={"latency_ms": 42.0})
    assert t.get("latency_ms") == 42.0


def test_telemetry_get_default():
    t = Telemetry()
    assert t.get("missing", 99.0) == 99.0


# ---------------------------------------------------------------------------
# AdaptationRule
# ---------------------------------------------------------------------------

def test_rule_not_triggered():
    rule = AdaptationRule(
        rule_id="r1",
        condition=lambda state: False,
        action=lambda state: "noop",
    )
    outcome = rule.evaluate({})
    assert not outcome.triggered


def test_rule_triggered_and_action_ok():
    calls: list[str] = []

    def action(state: dict) -> str:
        calls.append("executed")
        return "done"

    rule = AdaptationRule(
        rule_id="r2",
        condition=lambda state: True,
        action=action,
    )
    outcome = rule.evaluate({})
    assert outcome.triggered
    assert outcome.action_ok
    assert outcome.action_msg == "done"
    assert "executed" in calls


def test_rule_action_exception_captured():
    def bad_action(state):
        raise ValueError("broken")

    rule = AdaptationRule(
        rule_id="r3",
        condition=lambda state: True,
        action=bad_action,
    )
    outcome = rule.evaluate({})
    assert outcome.triggered
    assert not outcome.action_ok
    assert "broken" in outcome.action_msg


def test_rule_condition_exception_not_triggered():
    def bad_cond(state):
        raise RuntimeError("cond error")

    rule = AdaptationRule(
        rule_id="r4",
        condition=bad_cond,
        action=lambda s: "noop",
    )
    outcome = rule.evaluate({})
    assert not outcome.triggered
    assert "cond error" in outcome.action_msg


def test_rule_cooldown_blocks_retrigger():
    calls: list[int] = []

    rule = AdaptationRule(
        rule_id="cooldown",
        condition=lambda state: True,
        action=lambda state: calls.append(1) or "ok",
        cooldown_s=10.0,
    )
    rule.evaluate({})     # first trigger
    rule.evaluate({})     # should be blocked by cooldown
    assert len(calls) == 1


def test_rule_trigger_count():
    rule = AdaptationRule(
        rule_id="count-test",
        condition=lambda state: True,
        action=lambda state: "ok",
    )
    rule.evaluate({})
    rule.evaluate({})
    assert rule.trigger_count == 2


# ---------------------------------------------------------------------------
# CyberneticLoop — tick phases
# ---------------------------------------------------------------------------

def _simple_loop() -> CyberneticLoop:
    return CyberneticLoop(name="test-loop")


def test_tick_returns_summary():
    loop = _simple_loop()
    summary = loop.tick()
    assert "cycle" in summary
    assert "outcomes" in summary
    assert "safety" in summary
    assert summary["cycle"] == 0


def test_tick_increments_cycle():
    loop = _simple_loop()
    loop.tick()
    loop.tick()
    assert loop.cycle_count == 2


def test_sense_collects_telemetry():
    loop = _simple_loop()
    received: list[int] = []

    def src() -> Telemetry:
        received.append(1)
        return Telemetry(source="src1", metrics={"cpu": 0.5})

    loop.register_source(src)
    summary = loop.tick()
    assert summary["snapshots"] == 1
    assert len(received) == 1


def test_model_aggregates_metrics():
    loop = _simple_loop()
    loop.register_source(lambda: Telemetry(source="svc", metrics={"latency": 100.0}))
    loop.tick()
    assert "svc.latency" in loop.state


def test_model_running_average():
    loop = _simple_loop()
    loop.register_source(lambda: Telemetry(source="s", metrics={"x": 10.0}))
    loop.tick()
    loop.register_source(lambda: Telemetry(source="s", metrics={"x": 20.0}))
    loop.tick()
    # After two ticks the key exists and is a blend of 10 and 20
    assert "s.x" in loop.state
    assert 10.0 <= loop.state["s.x"] <= 20.0


def test_decide_and_act_rule_fires():
    loop = _simple_loop()
    fired: list[bool] = []
    rule = AdaptationRule(
        rule_id="fire-always",
        condition=lambda s: True,
        action=lambda s: fired.append(True) or "ok",
    )
    loop.register_rule(rule)
    summary = loop.tick()
    assert any(o["triggered"] for o in summary["outcomes"])
    assert len(fired) == 1


def test_verify_safety_check_fails():
    loop = CyberneticLoop(
        name="unsafe",
        safety_checks=[lambda state: False],   # always fail
    )
    summary = loop.tick()
    assert not summary["safety"]
    assert not summary["learned"]


def test_verify_safety_check_passes():
    loop = CyberneticLoop(
        name="safe",
        safety_checks=[lambda state: True],
    )
    summary = loop.tick()
    assert summary["safety"]
    assert summary["learned"]


def test_learn_increases_score_on_success():
    loop = _simple_loop()
    rule = AdaptationRule(
        rule_id="boost",
        condition=lambda s: True,
        action=lambda s: "ok",
        priority=1.0,
    )
    loop.register_rule(rule)
    initial_score = rule.score
    loop.tick()
    assert rule.score >= initial_score


def test_history_grows_each_tick():
    loop = _simple_loop()
    loop.tick()
    loop.tick()
    loop.tick()
    assert len(loop.history) == 3


def test_source_exception_silently_skipped():
    loop = _simple_loop()

    def bad_src():
        raise RuntimeError("sensor failure")

    loop.register_source(bad_src)
    summary = loop.tick()   # should not raise
    assert summary["snapshots"] == 0
