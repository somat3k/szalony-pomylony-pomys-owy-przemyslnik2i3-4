"""Plane D — Context Plane.

Models all relational state in the CROF fabric as a typed, directed graph.
Every entity — subject, claim, role, device, agent, module, event — is a
:class:`ContextNode`.  Relationships between them are :class:`ContextEdge`
instances with labelled types and optional metadata.

Visibility
----------
Each node and edge carries a *visibility class* aligned with Plane A:

* ``Open``        — readable by any authenticated actor
* ``Shared``      — readable within the same domain
* ``Restricted``  — requires the ``context.restricted.access`` capability
* ``Sealed``      — payload is encrypted; requires ``context.sealed.access``

Query model
-----------
The graph supports:

* ``get_node`` / ``get_edge``  — direct lookup by ID
* ``neighbours``               — all nodes reachable from a node
* ``predecessors``             — all nodes with an edge *into* a node
* ``query_by_type``            — filter nodes or edges by their type label
* ``subgraph``                 — extract a visibility-filtered subgraph
* ``path``                     — shortest path between two nodes (BFS)
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from hololang.crof.envelope import VisibilityClass


# ---------------------------------------------------------------------------
# Node types (extensible — add application-specific types as needed)
# ---------------------------------------------------------------------------

class NodeType:
    SUBJECT    = "subject"
    CLAIM      = "claim"
    ROLE       = "role"
    DEVICE     = "device"
    AGENT      = "agent"
    MODULE     = "module"
    EVENT      = "event"
    STATE      = "state"
    REQUEST    = "request"
    POLICY     = "policy"
    EVIDENCE   = "evidence"
    PROVENANCE = "provenance"
    DOMAIN     = "domain"
    SHARD      = "shard"


class EdgeType:
    HAS_CLAIM      = "has_claim"
    HAS_ROLE       = "has_role"
    OWNS_DEVICE    = "owns_device"
    RUNS_ON        = "runs_on"
    USES_MODULE    = "uses_module"
    PRODUCES_EVENT = "produces_event"
    ATTESTS        = "attests"
    GOVERNS        = "governs"
    DEPENDS_ON     = "depends_on"
    DERIVES_FROM   = "derives_from"
    OBSERVED_BY    = "observed_by"


# ---------------------------------------------------------------------------
# ContextNode
# ---------------------------------------------------------------------------

@dataclass
class ContextNode:
    """A typed node in the context graph.

    Parameters
    ----------
    node_id:
        Stable unique identifier.  Auto-generated if not provided.
    node_type:
        Semantic type (one of :class:`NodeType` constants or a custom string).
    label:
        Human-readable display label.
    visibility:
        :class:`~hololang.crof.envelope.VisibilityClass` string value.
    payload:
        Arbitrary structured data associated with this node.
    created_at:
        Unix epoch timestamp of creation.
    """
    node_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type:  str = NodeType.SUBJECT
    label:      str = ""
    visibility: str = VisibilityClass.OPEN.value
    payload:    dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"ContextNode({self.node_type}:{self.node_id[:8]}… {self.label!r})"


# ---------------------------------------------------------------------------
# ContextEdge
# ---------------------------------------------------------------------------

@dataclass
class ContextEdge:
    """A directed, typed edge between two :class:`ContextNode` objects.

    Parameters
    ----------
    source_id, target_id:
        Node IDs for the tail and head of the edge.
    edge_type:
        Semantic relationship type (one of :class:`EdgeType` constants).
    weight:
        Optional numeric weight (e.g. trust score, distance).
    metadata:
        Arbitrary structured data on the edge.
    """
    edge_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = EdgeType.HAS_CLAIM
    weight:    float = 1.0
    metadata:  dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))

    def __repr__(self) -> str:
        return (
            f"ContextEdge({self.source_id[:8]}… "
            f"--[{self.edge_type}]--> {self.target_id[:8]}…)"
        )


# ---------------------------------------------------------------------------
# ContextGraph  (Class-4 Context Node)
# ---------------------------------------------------------------------------

class ContextGraph:
    """Relational context graph for the CROF fabric.

    Stores :class:`ContextNode` and :class:`ContextEdge` objects and
    provides graph traversal, filtering, and serialisation utilities.
    """

    def __init__(self, name: str = "context") -> None:
        self.name:  str = name
        self._nodes: dict[str, ContextNode] = {}
        # adjacency list: node_id -> list of outgoing ContextEdge
        self._out:   dict[str, list[ContextEdge]] = {}
        # reverse adjacency: node_id -> list of incoming ContextEdge
        self._in:    dict[str, list[ContextEdge]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: ContextNode) -> ContextNode:
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])
        return node

    def add_edge(self, edge: ContextEdge) -> ContextEdge:
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id!r} not in graph")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id!r} not in graph")
        self._out[edge.source_id].append(edge)
        self._in[edge.target_id].append(edge)
        return edge

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges incident to it."""
        self._nodes.pop(node_id, None)
        # Drop outgoing edges
        for edge in self._out.pop(node_id, []):
            self._in.get(edge.target_id, [])
            self._in[edge.target_id] = [
                e for e in self._in.get(edge.target_id, [])
                if e.source_id != node_id
            ]
        # Drop incoming edges
        for edge in self._in.pop(node_id, []):
            self._out[edge.source_id] = [
                e for e in self._out.get(edge.source_id, [])
                if e.target_id != node_id
            ]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def make_node(
        self,
        node_type: str,
        label: str = "",
        visibility: str = VisibilityClass.OPEN.value,
        **payload: Any,
    ) -> ContextNode:
        node = ContextNode(node_type=node_type, label=label,
                           visibility=visibility, payload=dict(payload))
        return self.add_node(node)

    def connect(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        **metadata: Any,
    ) -> ContextEdge:
        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=dict(metadata),
        )
        return self.add_edge(edge)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ContextNode | None:
        return self._nodes.get(node_id)

    def require_node(self, node_id: str) -> ContextNode:
        n = self._nodes.get(node_id)
        if n is None:
            raise KeyError(f"Context node {node_id!r} not found")
        return n

    def neighbours(
        self, node_id: str, edge_type: str | None = None
    ) -> list[ContextNode]:
        """Return all nodes reachable via outgoing edges from *node_id*."""
        edges = self._out.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return [self._nodes[e.target_id] for e in edges if e.target_id in self._nodes]

    def predecessors(
        self, node_id: str, edge_type: str | None = None
    ) -> list[ContextNode]:
        """Return all nodes with an edge pointing *into* *node_id*."""
        edges = self._in.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return [self._nodes[e.source_id] for e in edges if e.source_id in self._nodes]

    def query_by_type(self, node_type: str) -> list[ContextNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def query_edges_by_type(self, edge_type: str) -> list[ContextEdge]:
        result: list[ContextEdge] = []
        for edges in self._out.values():
            result.extend(e for e in edges if e.edge_type == edge_type)
        return result

    # ------------------------------------------------------------------
    # Path finding (BFS)
    # ------------------------------------------------------------------

    def path(self, source_id: str, target_id: str) -> list[str] | None:
        """Return the shortest path of node IDs from *source_id* to *target_id*.

        Returns ``None`` if no path exists.
        """
        if source_id == target_id:
            return [source_id]
        visited: set[str] = {source_id}
        queue: deque[list[str]] = deque([[source_id]])
        while queue:
            current_path = queue.popleft()
            current = current_path[-1]
            for edge in self._out.get(current, []):
                nxt = edge.target_id
                if nxt == target_id:
                    return current_path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(current_path + [nxt])
        return None

    # ------------------------------------------------------------------
    # Visibility filtering
    # ------------------------------------------------------------------

    def subgraph(self, allowed_visibility: set[str]) -> "ContextGraph":
        """Return a new :class:`ContextGraph` containing only nodes whose
        visibility class is in *allowed_visibility*.
        """
        sub = ContextGraph(name=f"{self.name}.sub")
        for node in self._nodes.values():
            if node.visibility in allowed_visibility:
                sub.add_node(node)
        for edges in self._out.values():
            for edge in edges:
                if edge.source_id in sub._nodes and edge.target_id in sub._nodes:
                    sub._out[edge.source_id].append(edge)
                    sub._in[edge.target_id].append(edge)
        return sub

    # ------------------------------------------------------------------
    # Stats / serialisation
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._out.values())

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the context graph."""
        lines = [f'digraph "{self.name}" {{', '  rankdir=LR;']
        for node in self._nodes.values():
            label = f"{node.node_type}\\n{node.label or node.node_id[:8]}"
            lines.append(f'  "{node.node_id}" [label="{label}"];')
        for edges in self._out.values():
            for e in edges:
                lines.append(
                    f'  "{e.source_id}" -> "{e.target_id}" [label="{e.edge_type}"];'
                )
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ContextGraph(name={self.name!r}, "
            f"nodes={self.node_count}, edges={self.edge_count})"
        )
