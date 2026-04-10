"""Plane B — Identity Plane.

Manages every actor in the fabric:

* **Human** identities (operators, administrators)
* **Service** identities (microservices, gRPC endpoints)
* **Agent** identities (AI agents, automation scripts)
* **Device** identities (edge hardware, sensors)
* **Workload** identities (batch jobs, pipelines)

Each :class:`Subject` holds a stable ID, a cryptographic key, a set of
:class:`Capability` grants, and a history of :class:`RoleBinding` records.

The :class:`IdentityStore` is the in-process registry; in production it
would be backed by a distributed store (etcd, Spanner, etc.) fronted by the
:class:`~hololang.crof.fabric.CROFFabric`.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SubjectKind:
    HUMAN    = "human"
    SERVICE  = "service"
    AGENT    = "agent"
    DEVICE   = "device"
    WORKLOAD = "workload"
    MODULE   = "module"
    DOMAIN   = "domain"
    SHARD    = "shard"


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    """A named, scoped permission grant.

    Parameters
    ----------
    name:
        Human-readable capability label (e.g. ``"tensor.transform.execute"``).
    scope:
        Optional domain or resource scope (e.g. ``"domain:analytics"``).
    expires_at:
        Unix timestamp after which the capability is invalid.  ``None`` = permanent.
    metadata:
        Arbitrary extension fields.
    """
    name:       str
    scope:      str = ""
    expires_at: int | None = None
    metadata:   dict[str, Any] = field(default_factory=dict)

    def is_active(self, now: int | None = None) -> bool:
        t = now if now is not None else int(time.time())
        return self.expires_at is None or t <= self.expires_at

    def __repr__(self) -> str:
        return f"Capability({self.name!r}, scope={self.scope!r})"


# ---------------------------------------------------------------------------
# RoleBinding
# ---------------------------------------------------------------------------

@dataclass
class RoleBinding:
    """Binds a subject to a role within a scope."""
    subject_id: str
    role:       str
    scope:      str = ""
    granted_at: int = field(default_factory=lambda: int(time.time()))
    granted_by: str = ""

    def __repr__(self) -> str:
        return f"RoleBinding({self.subject_id!r} → {self.role!r})"


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

class Subject:
    """An authenticated actor in the CROF fabric.

    Parameters
    ----------
    subject_id:
        Stable, globally unique identifier.  Auto-generated if not provided.
    kind:
        One of the :class:`SubjectKind` constants.
    name:
        Human-readable display name.
    public_key:
        Optional bytes (e.g. DER-encoded public key).  CROF uses HMAC-based
        signatures internally; this field is reserved for external PKI.
    """

    def __init__(
        self,
        subject_id: str = "",
        kind: str = SubjectKind.SERVICE,
        name: str = "",
        public_key: bytes = b"",
    ) -> None:
        self.subject_id:    str            = subject_id or str(uuid.uuid4())
        self.kind:          str            = kind
        self.name:          str            = name or self.subject_id[:8]
        self.public_key:    bytes          = public_key
        # Derived signing secret (not stored in plain text in production)
        self._secret:       bytes          = secrets.token_bytes(32)
        self.capabilities:  list[Capability]  = []
        self.role_bindings: list[RoleBinding] = []
        self.metadata:      dict[str, Any] = {}
        self.created_at:    int            = int(time.time())
        self.revoked:       bool           = False

    # ------------------------------------------------------------------
    # Capability management
    # ------------------------------------------------------------------

    def grant_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def revoke_capability(self, name: str) -> None:
        self.capabilities = [c for c in self.capabilities if c.name != name]

    def has_capability(self, name: str, now: int | None = None) -> bool:
        return any(c.name == name and c.is_active(now) for c in self.capabilities)

    def active_capabilities(self, now: int | None = None) -> list[Capability]:
        return [c for c in self.capabilities if c.is_active(now)]

    # ------------------------------------------------------------------
    # Role bindings
    # ------------------------------------------------------------------

    def bind_role(self, role: str, scope: str = "", granted_by: str = "") -> RoleBinding:
        rb = RoleBinding(
            subject_id=self.subject_id,
            role=role,
            scope=scope,
            granted_by=granted_by,
        )
        self.role_bindings.append(rb)
        return rb

    def has_role(self, role: str, scope: str = "") -> bool:
        return any(
            rb.role == role and (not scope or rb.scope == scope)
            for rb in self.role_bindings
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id":    self.subject_id,
            "kind":          self.kind,
            "name":          self.name,
            "revoked":       self.revoked,
            "created_at":    self.created_at,
            "capabilities": [
                {"name": c.name, "scope": c.scope, "expires_at": c.expires_at}
                for c in self.capabilities
            ],
            "role_bindings": [
                {"role": rb.role, "scope": rb.scope}
                for rb in self.role_bindings
            ],
        }

    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the subject's public attributes."""
        raw = f"{self.subject_id}:{self.kind}:{self.name}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"Subject({self.subject_id[:8]}…, kind={self.kind!r}, name={self.name!r})"


# ---------------------------------------------------------------------------
# IdentityStore (Class-2 Identity Node)
# ---------------------------------------------------------------------------

class IdentityStore:
    """In-process registry of :class:`Subject` records.

    In a production deployment this would be backed by a distributed
    consistent store exposed via the CROF ``IdentityService`` gRPC interface.
    """

    def __init__(self, domain_id: str = "default") -> None:
        self.domain_id = domain_id
        self._subjects: dict[str, Subject] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, subject: Subject) -> Subject:
        """Add *subject* to the store.  Raises ``ValueError`` if already exists."""
        if subject.subject_id in self._subjects:
            raise ValueError(f"Subject {subject.subject_id!r} already registered")
        self._subjects[subject.subject_id] = subject
        return subject

    def get(self, subject_id: str) -> Subject | None:
        return self._subjects.get(subject_id)

    def require(self, subject_id: str) -> Subject:
        s = self._subjects.get(subject_id)
        if s is None:
            raise KeyError(f"Unknown subject {subject_id!r}")
        return s

    def revoke(self, subject_id: str) -> None:
        s = self.require(subject_id)
        s.revoked = True

    def remove(self, subject_id: str) -> None:
        self._subjects.pop(subject_id, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_subjects(self, kind: str | None = None) -> list[Subject]:
        subjects = list(self._subjects.values())
        if kind is not None:
            subjects = [s for s in subjects if s.kind == kind]
        return subjects

    def find_by_capability(self, cap_name: str) -> list[Subject]:
        return [
            s for s in self._subjects.values()
            if s.has_capability(cap_name) and not s.revoked
        ]

    def find_by_role(self, role: str, scope: str = "") -> list[Subject]:
        return [
            s for s in self._subjects.values()
            if s.has_role(role, scope) and not s.revoked
        ]

    @property
    def count(self) -> int:
        return len(self._subjects)

    def __repr__(self) -> str:
        return f"IdentityStore(domain={self.domain_id!r}, subjects={self.count})"
