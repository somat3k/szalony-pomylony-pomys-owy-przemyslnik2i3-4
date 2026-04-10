"""Plane C — Validation Plane.

Every :class:`~hololang.crof.envelope.Envelope` must pass four sequential
validation passes before being accepted by the core system:

Pass 1 — **Schema validation**
    The envelope's required fields are present and well-formed (non-empty
    IDs, valid visibility class, valid operation type, non-expired window).

Pass 2 — **Cryptographic validation**
    The envelope's ``signature`` field is verified against the actor's
    registered secret.  If the actor is unknown the envelope is rejected
    unless policy permits unsigned traffic.

Pass 3 — **Policy validation**
    The actor's claimed :class:`~hololang.crof.identity.Capability` is
    present in the :class:`~hololang.crof.identity.IdentityStore` and
    authorised for the requested :attr:`operation_type` within the envelope's
    :attr:`domain_id`.

Pass 4 — **Context validation**
    The operation is consistent with the visibility class:
    ``Sealed`` envelopes may only be dispatched by subjects with the
    ``context.sealed.access`` capability; ``Restricted`` envelopes require
    the ``context.restricted.access`` capability.

A :class:`ValidationResult` is returned for every envelope whether it
passes or fails.  The result carries per-pass flags and human-readable
messages for audit and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hololang.crof.envelope import Envelope, VisibilityClass, OperationType
from hololang.crof.identity import IdentityStore


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of the four-pass validation pipeline.

    Attributes
    ----------
    envelope_id:
        The ID of the validated envelope.
    passed:
        ``True`` only when all four passes succeed.
    schema_ok, crypto_ok, policy_ok, context_ok:
        Per-pass outcome flags.
    messages:
        Human-readable list of pass/failure notes (useful for audit logs).
    """
    envelope_id: str
    passed:      bool           = False
    schema_ok:   bool           = False
    crypto_ok:   bool           = False
    policy_ok:   bool           = False
    context_ok:  bool           = False
    messages:    list[str]      = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)

    def add(self, msg: str) -> None:
        self.messages.append(msg)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({status}, "
            f"schema={self.schema_ok}, crypto={self.crypto_ok}, "
            f"policy={self.policy_ok}, context={self.context_ok})"
        )


# ---------------------------------------------------------------------------
# Validator  (Class-3 Validation Node)
# ---------------------------------------------------------------------------

class Validator:
    """Orchestrates the four-pass validation pipeline for CROF envelopes.

    Parameters
    ----------
    identity_store:
        The :class:`~hololang.crof.identity.IdentityStore` used for actor
        lookup and capability checks.
    allow_unsigned:
        When ``True`` the cryptographic pass is skipped for actors whose
        signing secret is not available (useful in development/test mode).
    required_capabilities:
        Optional mapping of ``operation_type → required_capability_name``.
        Overrides the default capability rules.
    """

    # Default capability required per operation type
    _DEFAULT_CAP: dict[str, str] = {
        OperationType.QUERY.value:     "fabric.query",
        OperationType.COMMAND.value:   "fabric.command",
        OperationType.EVENT.value:     "fabric.event",
        OperationType.TRANSFORM.value: "fabric.transform",
        OperationType.ATTEST.value:    "fabric.attest",
    }

    # Capabilities required for sensitive visibility classes
    _VISIBILITY_CAP: dict[str, str] = {
        VisibilityClass.SEALED.value:     "context.sealed.access",
        VisibilityClass.RESTRICTED.value: "context.restricted.access",
    }

    def __init__(
        self,
        identity_store: IdentityStore,
        allow_unsigned: bool = False,
        required_capabilities: dict[str, str] | None = None,
    ) -> None:
        self._store        = identity_store
        self._allow_unsigned = allow_unsigned
        self._cap_map      = dict(self._DEFAULT_CAP)
        if required_capabilities:
            self._cap_map.update(required_capabilities)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        envelope: Envelope,
        secret_key: bytes | None = None,
        now: int | None = None,
    ) -> ValidationResult:
        """Run all four validation passes against *envelope*.

        Parameters
        ----------
        envelope:
            The envelope to validate.
        secret_key:
            Bytes used to verify HMAC-SHA256 signature.  If ``None``,
            the actor's stored ``_secret`` is used when available.
        now:
            Current Unix timestamp (injectable for deterministic testing).

        Returns
        -------
        ValidationResult
        """
        t   = now if now is not None else int(time.time())
        res = ValidationResult(envelope_id=envelope.envelope_id)

        # Pass 1: Schema
        res.schema_ok = self._pass_schema(envelope, t, res)
        if not res.schema_ok:
            res.passed = False
            return res

        # Pass 2: Cryptographic
        res.crypto_ok = self._pass_crypto(envelope, secret_key, res)
        if not res.crypto_ok and not self._allow_unsigned:
            res.passed = False
            return res
        if not res.crypto_ok and self._allow_unsigned:
            res.add("WARN: signature skipped (allow_unsigned=True)")
            res.crypto_ok = True  # treat as pass in permissive mode

        # Pass 3: Policy
        res.policy_ok = self._pass_policy(envelope, t, res)
        if not res.policy_ok:
            res.passed = False
            return res

        # Pass 4: Context
        res.context_ok = self._pass_context(envelope, t, res)
        res.passed = res.context_ok
        return res

    # ------------------------------------------------------------------
    # Individual passes
    # ------------------------------------------------------------------

    def _pass_schema(self, env: Envelope, t: int, res: ValidationResult) -> bool:
        ok = True
        if not env.envelope_id:
            res.add("SCHEMA: envelope_id is empty"); ok = False
        if not env.actor_id:
            res.add("SCHEMA: actor_id is empty"); ok = False
        if not env.domain_id:
            res.add("SCHEMA: domain_id is empty"); ok = False
        if env.visibility_class not in {v.value for v in VisibilityClass}:
            res.add(f"SCHEMA: unknown visibility_class {env.visibility_class!r}"); ok = False
        if env.operation_type not in {o.value for o in OperationType}:
            res.add(f"SCHEMA: unknown operation_type {env.operation_type!r}"); ok = False
        if not env.is_valid_window(t):
            res.add(
                f"SCHEMA: envelope window invalid "
                f"(issued_at={env.issued_at}, expires_at={env.expires_at}, now={t})"
            )
            ok = False
        if ok:
            res.add("SCHEMA: pass")
        return ok

    def _pass_crypto(
        self, env: Envelope, secret_key: bytes | None, res: ValidationResult
    ) -> bool:
        # Determine signing key
        key = secret_key
        if key is None:
            subject = self._store.get(env.actor_id)
            if subject is not None:
                key = subject._secret
        if key is None:
            if self._allow_unsigned:
                return False   # will be soft-passed by caller
            res.add("CRYPTO: no signing key available for actor")
            return False
        if not env.signature:
            res.add("CRYPTO: signature field is empty")
            return False
        if env.verify_signature(key):
            res.add("CRYPTO: pass")
            return True
        res.add("CRYPTO: signature mismatch")
        return False

    def _pass_policy(self, env: Envelope, t: int, res: ValidationResult) -> bool:
        subject = self._store.get(env.actor_id)
        if subject is None:
            res.add(f"POLICY: unknown actor {env.actor_id!r}")
            return False
        if subject.revoked:
            res.add(f"POLICY: actor {env.actor_id!r} is revoked")
            return False
        required_cap = self._cap_map.get(env.operation_type)
        if required_cap and not subject.has_capability(required_cap, t):
            res.add(
                f"POLICY: actor lacks capability {required_cap!r} "
                f"for operation {env.operation_type!r}"
            )
            return False
        res.add("POLICY: pass")
        return True

    def _pass_context(self, env: Envelope, t: int, res: ValidationResult) -> bool:
        vis_cap = self._VISIBILITY_CAP.get(env.visibility_class)
        if vis_cap is not None:
            subject = self._store.get(env.actor_id)
            if subject is None or not subject.has_capability(vis_cap, t):
                res.add(
                    f"CONTEXT: visibility {env.visibility_class!r} requires "
                    f"capability {vis_cap!r}"
                )
                return False
        res.add("CONTEXT: pass")
        return True
