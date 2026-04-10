"""Classified Relational Operations Fabric (CROF).

A seven-plane, gRPC-native distributed operating fabric implemented as
pure-Python modules.  Each plane is importable independently and can be
wired together via the :class:`~hololang.crof.fabric.CROFFabric` orchestrator.

Planes
------
A — Contract   : :mod:`hololang.crof.envelope`
B — Identity   : :mod:`hololang.crof.identity`
C — Validation : :mod:`hololang.crof.validation`
D — Context    : :mod:`hololang.crof.context`
E — Transform  : :mod:`hololang.crof.transform`
F — Interop    : :mod:`hololang.crof.interop`
G — Adaptation : :mod:`hololang.crof.adaptation`

Supporting
----------
Audit          : :mod:`hololang.crof.audit`
Fabric         : :mod:`hololang.crof.fabric`
"""

from hololang.crof.envelope   import Envelope, VisibilityClass, OperationType
from hololang.crof.identity   import Subject, Capability, IdentityStore
from hololang.crof.validation import Validator, ValidationResult
from hololang.crof.context    import ContextGraph, ContextNode, ContextEdge
from hololang.crof.transform  import TransformModule, TransformGraph, TransformResult
from hololang.crof.interop    import DomainAdapter, LayerZeroAdapter, GrpcAdapter
from hololang.crof.adaptation import CyberneticLoop, AdaptationRule, Telemetry
from hololang.crof.audit      import AuditLedger, AuditEvent
from hololang.crof.fabric     import CROFFabric

__all__ = [
    "Envelope", "VisibilityClass", "OperationType",
    "Subject", "Capability", "IdentityStore",
    "Validator", "ValidationResult",
    "ContextGraph", "ContextNode", "ContextEdge",
    "TransformModule", "TransformGraph", "TransformResult",
    "DomainAdapter", "LayerZeroAdapter", "GrpcAdapter",
    "CyberneticLoop", "AdaptationRule", "Telemetry",
    "AuditLedger", "AuditEvent",
    "CROFFabric",
]
