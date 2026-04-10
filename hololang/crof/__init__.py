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
E5 — Compute   : :mod:`hololang.crof.compute`   (Class-5 TensorComputeNode)
F — Interop    : :mod:`hololang.crof.interop`
G — Adaptation : :mod:`hololang.crof.adaptation`

Supporting
----------
Audit          : :mod:`hololang.crof.audit`
Receiver       : :mod:`hololang.crof.receiver`   (aspect-based envelope routing)
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
from hololang.crof.receiver   import Receiver, ReceiverChain, ReceiverDecision
from hololang.crof.compute    import (
    TFBackend,
    MockTFBackend,
    TensorFlowBackend,
    ModelDeployment,
    ModelOutput,
    TensorComputeNode,
)

__all__ = [
    # Plane A — Contract
    "Envelope", "VisibilityClass", "OperationType",
    # Plane B — Identity
    "Subject", "Capability", "IdentityStore",
    # Plane C — Validation
    "Validator", "ValidationResult",
    # Plane D — Context
    "ContextGraph", "ContextNode", "ContextEdge",
    # Plane E — Transform
    "TransformModule", "TransformGraph", "TransformResult",
    # Plane E5 — Tensor Compute Node
    "TFBackend", "MockTFBackend", "TensorFlowBackend",
    "ModelDeployment", "ModelOutput", "TensorComputeNode",
    # Plane F — Interop
    "DomainAdapter", "LayerZeroAdapter", "GrpcAdapter",
    # Plane G — Adaptation
    "CyberneticLoop", "AdaptationRule", "Telemetry",
    # Supporting
    "AuditLedger", "AuditEvent",
    "Receiver", "ReceiverChain", "ReceiverDecision",
    "CROFFabric",
]
