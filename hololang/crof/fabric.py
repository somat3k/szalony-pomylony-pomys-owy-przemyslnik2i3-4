"""CROF Fabric — top-level orchestrator.

:class:`CROFFabric` wires all seven planes together into a single
cohesive runtime.  It provides the unified API surface described in the
CROF blueprint's §5 "Core gRPC service surface" — implemented here as
Python method calls rather than network RPCs, making it embeddable directly
inside HoloLang programs and tests.

A fabric instance acts as both a **Class-1 Root Governance Node** (policy,
schema, deployment rules) and an in-process **domain hub** that delegates
to the individual planes.

Quick start::

    from hololang.crof import CROFFabric
    from hololang.crof.envelope import Envelope, OperationType
    from hololang.crof.identity import Subject, Capability, SubjectKind

    fabric = CROFFabric("my-domain")

    # Register an actor
    svc = Subject(kind=SubjectKind.SERVICE, name="analytics-svc")
    svc.grant_capability(Capability("fabric.transform"))
    fabric.identity.register(svc)

    # Build and dispatch an envelope
    env = Envelope.build(
        actor_id="",          # filled below
        domain_id="my-domain",
        operation_type=OperationType.TRANSFORM.value,
    )
    env.actor_id = svc.subject_id
    env = env.sign(svc._secret)

    result = fabric.dispatch(env)
    print(result)
"""

from __future__ import annotations

from typing import Any, Callable

from hololang.crof.envelope   import Envelope, OperationType
from hololang.crof.identity   import IdentityStore, Subject, Capability
from hololang.crof.validation import Validator, ValidationResult
from hololang.crof.context    import ContextGraph
from hololang.crof.transform  import TransformGraph, TransformResult
from hololang.crof.interop    import DomainAdapter, DispatchResult
from hololang.crof.adaptation import CyberneticLoop, AdaptationRule, Telemetry
from hololang.crof.audit      import AuditLedger, AuditEvent


class CROFFabric:
    """Unified CROF runtime that binds all seven planes.

    Parameters
    ----------
    domain_id:
        The fabric's own domain identifier.
    allow_unsigned:
        Pass to :class:`~hololang.crof.validation.Validator`; useful in
        development mode.
    """

    def __init__(
        self,
        domain_id: str = "default",
        allow_unsigned: bool = False,
    ) -> None:
        self.domain_id = domain_id

        # Plane B — Identity
        self.identity   = IdentityStore(domain_id=domain_id)
        # Plane C — Validation
        self.validator  = Validator(self.identity, allow_unsigned=allow_unsigned)
        # Plane D — Context
        self.context    = ContextGraph(name=domain_id)
        # Plane E — Transform
        self.transforms = TransformGraph(name=f"{domain_id}.transforms")
        # Plane F — Interop
        self._adapters: dict[str, DomainAdapter] = {}
        # Plane G — Adaptation
        self.loop       = CyberneticLoop(name=f"{domain_id}.loop")
        # Audit
        self.audit      = AuditLedger(name=domain_id)

    # ------------------------------------------------------------------
    # Plane F — Interop / adapter management
    # ------------------------------------------------------------------

    def register_adapter(self, name: str, adapter: DomainAdapter) -> None:
        """Register a domain adapter under *name*."""
        self._adapters[name] = adapter

    def get_adapter(self, name: str) -> DomainAdapter | None:
        return self._adapters.get(name)

    # ------------------------------------------------------------------
    # Core service surface
    # ------------------------------------------------------------------

    def dispatch(
        self,
        envelope: Envelope,
        adapter_name: str | None = None,
        validate: bool = True,
    ) -> DispatchResult | ValidationResult:
        """Validate and dispatch *envelope*.

        If *validate* is ``True`` (default), the envelope is run through
        the four-pass :class:`~hololang.crof.validation.Validator`.
        Rejected envelopes are returned as a :class:`~hololang.crof.validation.ValidationResult`
        without being dispatched.

        If *adapter_name* is provided, the envelope is forwarded to that
        adapter.  Otherwise it is processed locally by the transform graph.

        Returns
        -------
        ValidationResult | DispatchResult
        """
        if validate:
            vr = self.validator.validate(envelope)
            if not vr.passed:
                self.audit.emit(
                    "envelope.rejected",
                    source   = "fabric",
                    actor_id = envelope.actor_id,
                    domain_id= self.domain_id,
                    reason   = "; ".join(vr.messages),
                )
                return vr
            self.audit.emit(
                "envelope.validated",
                source    = "fabric",
                actor_id  = envelope.actor_id,
                domain_id = self.domain_id,
            )

        if adapter_name is not None:
            adapter = self._adapters.get(adapter_name)
            if adapter is None:
                raise KeyError(f"No adapter registered as {adapter_name!r}")
            result = adapter.dispatch(envelope)
            self.audit.emit(
                "envelope.dispatched",
                source       = "fabric",
                actor_id     = envelope.actor_id,
                domain_id    = self.domain_id,
                adapter      = adapter_name,
                status       = result.status,
            )
            return result

        # Local dispatch: run through the transform graph
        ctx: dict[str, Any] = {}
        try:
            self.transforms.run(envelope, ctx)
            status = "delivered"
        except Exception as exc:  # noqa: BLE001
            status = f"error:{exc}"

        self.audit.emit(
            "envelope.dispatched",
            source    = "fabric",
            actor_id  = envelope.actor_id,
            domain_id = self.domain_id,
            status    = status,
        )
        from hololang.crof.interop import DispatchResult as DR
        return DR(
            adapter_type = "local",
            status       = "delivered" if status == "delivered" else "failed",
            error        = "" if status == "delivered" else status,
        )

    def authorize(
        self,
        actor_id:    str,
        capability:  str,
        domain_id:   str | None = None,
    ) -> bool:
        """Check whether *actor_id* holds *capability*.

        Parameters
        ----------
        actor_id:
            Subject to check.
        capability:
            Required capability name.
        domain_id:
            If provided, the subject must also have a role in this domain
            (currently unused — reserved for future scope binding).

        Returns
        -------
        bool
        """
        subject = self.identity.get(actor_id)
        if subject is None or subject.revoked:
            return False
        ok = subject.has_capability(capability)
        self.audit.emit(
            "policy.authorized" if ok else "policy.denied",
            source    = "fabric",
            actor_id  = actor_id,
            domain_id = self.domain_id,
            capability= capability,
        )
        return ok

    def tick(self) -> dict[str, Any]:
        """Run one cybernetic adaptation cycle and return its summary."""
        summary = self.loop.tick()
        self.audit.emit(
            "adaptation.triggered",
            source    = "fabric",
            domain_id = self.domain_id,
            cycle     = summary.get("cycle"),
            triggered = sum(1 for o in summary.get("outcomes", []) if o["triggered"]),
        )
        return summary

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    def register_telemetry_source(self, source: Callable[[], Telemetry]) -> None:
        self.loop.register_source(source)

    def register_rule(self, rule: AdaptationRule) -> None:
        self.loop.register_rule(rule)

    def __repr__(self) -> str:
        return (
            f"CROFFabric(domain={self.domain_id!r}, "
            f"subjects={self.identity.count}, "
            f"audited={self.audit.count})"
        )
