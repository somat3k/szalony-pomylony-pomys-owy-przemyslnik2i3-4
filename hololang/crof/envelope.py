"""Plane A — Contract Plane: Universal Envelope.

Every CROF message travels inside an :class:`Envelope`.  The envelope
carries full identity, authority, scope, context, policy, provenance, and
replay-protection fields.  Its structure is intentionally protobuf-aligned
so that the dataclass can be mechanically mapped to a ``.proto`` schema
for production gRPC transport.

Visibility classes
------------------
``Open``        — readable by any authenticated actor
``Shared``      — readable within the same domain or tenant
``Restricted``  — readable only by explicitly named subjects
``Sealed``      — opaque; payload is encrypted; readable only by holder

Operation types
---------------
``Query``       — read-only information request
``Command``     — state-mutating instruction
``Event``       — notification of something that occurred
``Transform``   — request to run a tensorified module
``Attest``      — cryptographic attestation submission
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations (proto-compatible string values)
# ---------------------------------------------------------------------------

class VisibilityClass(str, Enum):
    OPEN       = "Open"
    SHARED     = "Shared"
    RESTRICTED = "Restricted"
    SEALED     = "Sealed"


class OperationType(str, Enum):
    QUERY     = "Query"
    COMMAND   = "Command"
    EVENT     = "Event"
    TRANSFORM = "Transform"
    ATTEST    = "Attest"


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """Universal message envelope for all CROF inter-component traffic.

    Parameters mirror the protobuf schema so that serialisation to/from
    ``dict`` or JSON is lossless and binary-codec-friendly.

    Fields
    ------
    envelope_id     : unique message identifier (UUIDv4)
    tenant_id       : isolation boundary (multi-tenancy)
    domain_id       : logical domain (service cluster, chain, edge node …)
    subject_id      : the subject *about whom* this message is sent
    actor_id        : the principal *sending* the message
    module_id       : the transform module referenced
    capability_id   : the capability claimed by the actor
    policy_id       : the policy governing this operation
    context_id      : opaque reference to a context snapshot
    visibility_class: ``Open | Shared | Restricted | Sealed``
    operation_type  : ``Query | Command | Event | Transform | Attest``
    payload_type    : MIME-type or semantic label for ``payload``
    payload         : raw bytes (application-level message body)
    evidence_refs   : references to cryptographic proofs or signed claims
    provenance_refs : lineage chain references (prior envelope IDs)
    signature       : HMAC-SHA256 over canonical representation
    issued_at       : Unix epoch (seconds) when the envelope was created
    expires_at      : Unix epoch (seconds) after which the envelope is invalid
    """

    envelope_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id:        str = ""
    domain_id:        str = ""
    subject_id:       str = ""
    actor_id:         str = ""
    module_id:        str = ""
    capability_id:    str = ""
    policy_id:        str = ""
    context_id:       str = ""
    visibility_class: str = VisibilityClass.OPEN.value
    operation_type:   str = OperationType.QUERY.value
    payload_type:     str = "application/octet-stream"
    payload:          bytes = field(default_factory=bytes)
    evidence_refs:    list[str] = field(default_factory=list)
    provenance_refs:  list[str] = field(default_factory=list)
    signature:        bytes = field(default_factory=bytes)
    issued_at:        int = field(default_factory=lambda: int(time.time()))
    expires_at:       int = field(default_factory=lambda: int(time.time()) + 3600)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id":      self.envelope_id,
            "tenant_id":        self.tenant_id,
            "domain_id":        self.domain_id,
            "subject_id":       self.subject_id,
            "actor_id":         self.actor_id,
            "module_id":        self.module_id,
            "capability_id":    self.capability_id,
            "policy_id":        self.policy_id,
            "context_id":       self.context_id,
            "visibility_class": self.visibility_class,
            "operation_type":   self.operation_type,
            "payload_type":     self.payload_type,
            "payload":          self.payload.hex(),
            "evidence_refs":    list(self.evidence_refs),
            "provenance_refs":  list(self.provenance_refs),
            "signature":        self.signature.hex(),
            "issued_at":        self.issued_at,
            "expires_at":       self.expires_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Envelope":
        return cls(
            envelope_id      = d.get("envelope_id", str(uuid.uuid4())),
            tenant_id        = d.get("tenant_id", ""),
            domain_id        = d.get("domain_id", ""),
            subject_id       = d.get("subject_id", ""),
            actor_id         = d.get("actor_id", ""),
            module_id        = d.get("module_id", ""),
            capability_id    = d.get("capability_id", ""),
            policy_id        = d.get("policy_id", ""),
            context_id       = d.get("context_id", ""),
            visibility_class = d.get("visibility_class", VisibilityClass.OPEN.value),
            operation_type   = d.get("operation_type", OperationType.QUERY.value),
            payload_type     = d.get("payload_type", "application/octet-stream"),
            payload          = bytes.fromhex(d.get("payload", "")),
            evidence_refs    = list(d.get("evidence_refs", [])),
            provenance_refs  = list(d.get("provenance_refs", [])),
            signature        = bytes.fromhex(d.get("signature", "")),
            issued_at        = d.get("issued_at", int(time.time())),
            expires_at       = d.get("expires_at", int(time.time()) + 3600),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Envelope":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Canonical bytes (used for signing — excludes the signature field)
    # ------------------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """Return a deterministic byte representation for signing."""
        d = self.to_dict()
        d.pop("signature")
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

    # ------------------------------------------------------------------
    # Signing / verification
    # ------------------------------------------------------------------

    def sign(self, secret_key: bytes) -> "Envelope":
        """Return a new :class:`Envelope` with an HMAC-SHA256 signature."""
        sig = hmac.new(secret_key, self.canonical_bytes(), hashlib.sha256).digest()
        import dataclasses
        return dataclasses.replace(self, signature=sig)

    def verify_signature(self, secret_key: bytes) -> bool:
        """Verify the HMAC-SHA256 signature against *secret_key*."""
        expected = hmac.new(secret_key, self.canonical_bytes(), hashlib.sha256).digest()
        return hmac.compare_digest(expected, self.signature)

    # ------------------------------------------------------------------
    # Temporal validity
    # ------------------------------------------------------------------

    def is_expired(self, now: int | None = None) -> bool:
        """Return ``True`` if the envelope has passed its ``expires_at``."""
        t = now if now is not None else int(time.time())
        return t > self.expires_at

    def is_valid_window(self, now: int | None = None) -> bool:
        """Return ``True`` when ``issued_at <= now <= expires_at``."""
        t = now if now is not None else int(time.time())
        return self.issued_at <= t <= self.expires_at

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        actor_id: str,
        domain_id: str,
        payload: bytes = b"",
        operation_type: str = OperationType.QUERY.value,
        visibility_class: str = VisibilityClass.OPEN.value,
        ttl_seconds: int = 3600,
        **kwargs: Any,
    ) -> "Envelope":
        """Convenience constructor with sensible defaults."""
        now = int(time.time())
        return cls(
            actor_id         = actor_id,
            domain_id        = domain_id,
            payload          = payload,
            operation_type   = operation_type,
            visibility_class = visibility_class,
            issued_at        = now,
            expires_at       = now + ttl_seconds,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"Envelope(id={self.envelope_id[:8]}…, op={self.operation_type}, "
            f"actor={self.actor_id!r}, domain={self.domain_id!r})"
        )
