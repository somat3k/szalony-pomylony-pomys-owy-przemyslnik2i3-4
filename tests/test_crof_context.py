"""Tests for CROF Context Plane (Plane D) — relational graph."""

import pytest
from hololang.crof.context import (
    ContextGraph, ContextNode, ContextEdge, NodeType, EdgeType,
)
from hololang.crof.envelope import VisibilityClass


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def test_add_and_get_node():
    g = ContextGraph("test")
    n = ContextNode(node_type=NodeType.SUBJECT, label="Alice")
    g.add_node(n)
    assert g.get_node(n.node_id) is n


def test_require_node_missing():
    g = ContextGraph()
    with pytest.raises(KeyError):
        g.require_node("non-existent")


def test_remove_node():
    g = ContextGraph()
    n = g.make_node(NodeType.CLAIM, label="some-claim")
    g.remove_node(n.node_id)
    assert g.get_node(n.node_id) is None


def test_node_count():
    g = ContextGraph()
    assert g.node_count == 0
    g.make_node(NodeType.SUBJECT)
    assert g.node_count == 1


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def test_add_edge():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT, label="A")
    b = g.make_node(NodeType.ROLE, label="admin")
    edge = g.connect(a.node_id, b.node_id, EdgeType.HAS_ROLE)
    assert edge.source_id == a.node_id
    assert edge.target_id == b.node_id


def test_edge_unknown_source_raises():
    g = ContextGraph()
    b = g.make_node(NodeType.ROLE)
    with pytest.raises(KeyError):
        g.connect("missing", b.node_id, EdgeType.HAS_ROLE)


def test_edge_unknown_target_raises():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    with pytest.raises(KeyError):
        g.connect(a.node_id, "missing", EdgeType.HAS_CLAIM)


def test_edge_count():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    b = g.make_node(NodeType.CLAIM)
    g.connect(a.node_id, b.node_id, EdgeType.HAS_CLAIM)
    assert g.edge_count == 1


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def test_neighbours():
    g = ContextGraph()
    s = g.make_node(NodeType.SUBJECT, label="S")
    r = g.make_node(NodeType.ROLE, label="R")
    d = g.make_node(NodeType.DEVICE, label="D")
    g.connect(s.node_id, r.node_id, EdgeType.HAS_ROLE)
    g.connect(s.node_id, d.node_id, EdgeType.OWNS_DEVICE)
    nbrs = g.neighbours(s.node_id)
    assert len(nbrs) == 2


def test_neighbours_by_type():
    g = ContextGraph()
    s = g.make_node(NodeType.SUBJECT)
    r = g.make_node(NodeType.ROLE)
    d = g.make_node(NodeType.DEVICE)
    g.connect(s.node_id, r.node_id, EdgeType.HAS_ROLE)
    g.connect(s.node_id, d.node_id, EdgeType.OWNS_DEVICE)
    roles = g.neighbours(s.node_id, edge_type=EdgeType.HAS_ROLE)
    assert len(roles) == 1
    assert roles[0].node_id == r.node_id


def test_predecessors():
    g = ContextGraph()
    s = g.make_node(NodeType.SUBJECT)
    r = g.make_node(NodeType.ROLE)
    g.connect(s.node_id, r.node_id, EdgeType.HAS_ROLE)
    preds = g.predecessors(r.node_id)
    assert len(preds) == 1
    assert preds[0].node_id == s.node_id


def test_query_by_type():
    g = ContextGraph()
    g.make_node(NodeType.SUBJECT)
    g.make_node(NodeType.SUBJECT)
    g.make_node(NodeType.CLAIM)
    subjects = g.query_by_type(NodeType.SUBJECT)
    assert len(subjects) == 2


def test_query_edges_by_type():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    b = g.make_node(NodeType.CLAIM)
    c = g.make_node(NodeType.ROLE)
    g.connect(a.node_id, b.node_id, EdgeType.HAS_CLAIM)
    g.connect(a.node_id, c.node_id, EdgeType.HAS_ROLE)
    claims = g.query_edges_by_type(EdgeType.HAS_CLAIM)
    assert len(claims) == 1


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

def test_path_direct():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT, label="A")
    b = g.make_node(NodeType.SUBJECT, label="B")
    g.connect(a.node_id, b.node_id, EdgeType.DEPENDS_ON)
    p = g.path(a.node_id, b.node_id)
    assert p == [a.node_id, b.node_id]


def test_path_indirect():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    b = g.make_node(NodeType.SUBJECT)
    c = g.make_node(NodeType.SUBJECT)
    g.connect(a.node_id, b.node_id, EdgeType.DEPENDS_ON)
    g.connect(b.node_id, c.node_id, EdgeType.DEPENDS_ON)
    p = g.path(a.node_id, c.node_id)
    assert p is not None
    assert p[0] == a.node_id
    assert p[-1] == c.node_id


def test_path_none_when_disconnected():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    b = g.make_node(NodeType.SUBJECT)
    assert g.path(a.node_id, b.node_id) is None


def test_path_self():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT)
    p = g.path(a.node_id, a.node_id)
    assert p == [a.node_id]


# ---------------------------------------------------------------------------
# Visibility subgraph
# ---------------------------------------------------------------------------

def test_subgraph_filters_visibility():
    g = ContextGraph()
    pub = g.make_node(NodeType.SUBJECT, visibility=VisibilityClass.OPEN.value)
    priv = g.make_node(NodeType.SUBJECT, visibility=VisibilityClass.SEALED.value)
    sub = g.subgraph({VisibilityClass.OPEN.value})
    assert sub.get_node(pub.node_id) is not None
    assert sub.get_node(priv.node_id) is None


def test_subgraph_includes_connected_edges():
    g = ContextGraph()
    a = g.make_node(NodeType.SUBJECT, visibility=VisibilityClass.OPEN.value)
    b = g.make_node(NodeType.CLAIM, visibility=VisibilityClass.OPEN.value)
    g.connect(a.node_id, b.node_id, EdgeType.HAS_CLAIM)
    sub = g.subgraph({VisibilityClass.OPEN.value})
    assert sub.edge_count == 1


# ---------------------------------------------------------------------------
# DOT export
# ---------------------------------------------------------------------------

def test_to_dot():
    g = ContextGraph("mygraph")
    a = g.make_node(NodeType.SUBJECT, label="A")
    b = g.make_node(NodeType.ROLE, label="B")
    g.connect(a.node_id, b.node_id, EdgeType.HAS_ROLE)
    dot = g.to_dot()
    assert 'digraph "mygraph"' in dot
    assert "has_role" in dot
