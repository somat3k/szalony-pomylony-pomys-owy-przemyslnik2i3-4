"""Plane G — Adaptation Plane: cybernetic feedback loop.

The adaptation plane implements the *sense → model → decide → act →
verify → learn* cybernetic control loop described in the CROF blueprint.

Architecture
------------
:class:`Telemetry`
    Metrics collected from services, devices, chains, and transforms.
    Each snapshot is a dict of scalar measurements keyed by metric name.

:class:`AdaptationRule`
    A condition + action pair evaluated each cycle.  If the condition
    returns ``True`` for the current telemetry snapshot, the action is
    executed and its outcome is recorded.

:class:`CyberneticLoop`
    Orchestrates the full six-phase control loop:

    1. **Sense**   — collect a :class:`Telemetry` snapshot from registered sources
    2. **Model**   — aggregate and summarise the snapshot into a live state view
    3. **Decide**  — evaluate all :class:`AdaptationRule` conditions
    4. **Act**     — execute triggered rules and record outcomes
    5. **Verify**  — check integrity and safety bounds on the new state
    6. **Learn**   — update routing weights and rule priority scores

The loop runs synchronously via :meth:`CyberneticLoop.tick`; in production
it would be driven by a background thread or an async scheduler.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """A single telemetry snapshot.

    Attributes
    ----------
    snapshot_id:
        Auto-generated unique identifier for this snapshot.
    source:
        Identifier of the component that produced the snapshot
        (e.g. service name, device ID, chain endpoint).
    metrics:
        Key → value mapping of scalar measurements.
    timestamp:
        Unix epoch (seconds) when the snapshot was collected.
    tags:
        Optional labels (environment, region, trust class …).
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source:      str = ""
    metrics:     dict[str, float] = field(default_factory=dict)
    timestamp:   int = field(default_factory=lambda: int(time.time()))
    tags:        dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.metrics.get(key, default)

    def __repr__(self) -> str:
        return f"Telemetry(source={self.source!r}, metrics={len(self.metrics)})"


# ---------------------------------------------------------------------------
# AdaptationRule
# ---------------------------------------------------------------------------

@dataclass
class RuleOutcome:
    """Result of evaluating and executing an :class:`AdaptationRule`."""
    rule_id:     str
    triggered:   bool
    action_ok:   bool  = False
    action_msg:  str   = ""
    timestamp:   int   = field(default_factory=lambda: int(time.time()))


class AdaptationRule:
    """A condition + action pair evaluated each adaptation cycle.

    Parameters
    ----------
    rule_id:
        Unique identifier.
    condition:
        Callable ``(state: dict[str, Any]) -> bool``.  Return ``True`` to
        trigger the action.
    action:
        Callable ``(state: dict[str, Any]) -> str`` that modifies system
        state and returns a human-readable outcome message.
    priority:
        Numeric priority (higher = evaluated first).
    description:
        Human-readable description of the rule's intent.
    cooldown_s:
        Minimum seconds between successive triggers of this rule.
    """

    def __init__(
        self,
        rule_id:     str,
        condition:   Callable[[dict[str, Any]], bool],
        action:      Callable[[dict[str, Any]], str],
        priority:    float = 0.0,
        description: str = "",
        cooldown_s:  float = 0.0,
    ) -> None:
        self.rule_id     = rule_id
        self.condition   = condition
        self.action      = action
        self.priority    = priority
        self.description = description
        self.cooldown_s  = cooldown_s
        self._last_triggered: float = 0.0
        self.trigger_count:   int   = 0
        self.score:           float = priority  # adaptive priority score

    def evaluate(self, state: dict[str, Any]) -> RuleOutcome:
        now = time.perf_counter()
        if self.cooldown_s > 0 and (now - self._last_triggered) < self.cooldown_s:
            return RuleOutcome(rule_id=self.rule_id, triggered=False,
                               action_msg="cooldown")
        try:
            triggered = bool(self.condition(state))
        except Exception as exc:  # noqa: BLE001
            return RuleOutcome(rule_id=self.rule_id, triggered=False,
                               action_msg=f"condition error: {exc}")
        if not triggered:
            return RuleOutcome(rule_id=self.rule_id, triggered=False)
        # Execute action
        try:
            msg = self.action(state)
            ok  = True
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            ok  = False
        self._last_triggered = now
        self.trigger_count  += 1
        return RuleOutcome(rule_id=self.rule_id, triggered=True,
                           action_ok=ok, action_msg=msg or "")

    def __repr__(self) -> str:
        return (
            f"AdaptationRule({self.rule_id!r}, priority={self.priority:.1f}, "
            f"triggers={self.trigger_count})"
        )


# ---------------------------------------------------------------------------
# CyberneticLoop
# ---------------------------------------------------------------------------

class CyberneticLoop:
    """Orchestrates the CROF cybernetic adaptation loop.

    Parameters
    ----------
    name:
        Human-readable loop identifier.
    safety_checks:
        Optional list of callables ``(state) -> bool``.  If any returns
        ``False`` during the *verify* phase, the loop records a safety
        violation and skips *learn*.
    """

    def __init__(
        self,
        name: str = "crof-adaptation",
        safety_checks: list[Callable[[dict[str, Any]], bool]] | None = None,
    ) -> None:
        self.name           = name
        self._rules:        list[AdaptationRule] = []
        self._sources:      list[Callable[[], Telemetry]] = []
        self._safety_checks = safety_checks or []
        self._state:        dict[str, Any] = {}
        self._history:      list[dict[str, Any]] = []   # per-tick summary
        self.cycle_count:   int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_rule(self, rule: AdaptationRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.score, reverse=True)

    def register_source(self, source: Callable[[], Telemetry]) -> None:
        self._sources.append(source)

    # ------------------------------------------------------------------
    # The six-phase tick
    # ------------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """Run one full adaptation cycle.

        Returns a summary dict with per-phase outcomes.
        """
        summary: dict[str, Any] = {
            "cycle":    self.cycle_count,
            "outcomes": [],
            "safety":   True,
            "learned":  False,
        }

        # Phase 1 — Sense
        snapshots = self._sense()
        summary["snapshots"] = len(snapshots)

        # Phase 2 — Model
        self._model(snapshots)
        summary["state_keys"] = list(self._state.keys())

        # Phase 3 — Decide + Phase 4 — Act
        outcomes = self._decide_and_act()
        summary["outcomes"] = [
            {"rule": o.rule_id, "triggered": o.triggered, "ok": o.action_ok}
            for o in outcomes
        ]

        # Phase 5 — Verify
        safe = self._verify()
        summary["safety"] = safe

        # Phase 6 — Learn (skip on safety violation)
        if safe:
            self._learn(outcomes)
            summary["learned"] = True

        self._history.append(summary)
        self.cycle_count += 1
        return summary

    # ------------------------------------------------------------------
    # Individual phases
    # ------------------------------------------------------------------

    def _sense(self) -> list[Telemetry]:
        snapshots: list[Telemetry] = []
        for src in self._sources:
            try:
                snap = src()
                snapshots.append(snap)
            except Exception:  # noqa: BLE001
                pass
        return snapshots

    def _model(self, snapshots: list[Telemetry]) -> None:
        """Aggregate snapshots into ``self._state``."""
        self._state["__snapshots__"] = len(snapshots)
        self._state["__timestamp__"] = int(time.time())
        for snap in snapshots:
            for k, v in snap.metrics.items():
                key = f"{snap.source}.{k}"
                # Running average
                if key in self._state:
                    self._state[key] = 0.5 * self._state[key] + 0.5 * v
                else:
                    self._state[key] = v

    def _decide_and_act(self) -> list[RuleOutcome]:
        outcomes: list[RuleOutcome] = []
        for rule in self._rules:
            outcome = rule.evaluate(self._state)
            outcomes.append(outcome)
        return outcomes

    def _verify(self) -> bool:
        for check in self._safety_checks:
            try:
                if not check(self._state):
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def _learn(self, outcomes: list[RuleOutcome]) -> None:
        """Update rule priority scores based on action success rate."""
        for outcome in outcomes:
            for rule in self._rules:
                if rule.rule_id == outcome.rule_id and outcome.triggered:
                    # Increase score on success, decrease on failure
                    delta = 0.1 if outcome.action_ok else -0.1
                    rule.score = max(0.0, rule.score + delta)
        # Re-sort by updated scores
        self._rules.sort(key=lambda r: r.score, reverse=True)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"CyberneticLoop(name={self.name!r}, "
            f"rules={len(self._rules)}, cycles={self.cycle_count})"
        )
