"""Tests for GAS embeddings (tensor/embeddings.py)."""

import math
import pytest
from hololang.tensor.tensor import Tensor
from hololang.tensor.embeddings import EmbeddingSpace


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

def test_set_and_get():
    space = EmbeddingSpace(dim=3)
    space.set("cat", [1.0, 0.0, 0.0])
    vec = space.get("cat")
    assert vec is not None
    assert vec._data == [1.0, 0.0, 0.0]


def test_set_from_tensor():
    space = EmbeddingSpace(dim=2)
    t = Tensor([2], [0.5, -0.5])
    space.set("dog", t)
    assert space.get("dog")._data == [0.5, -0.5]


def test_dim_mismatch_raises():
    space = EmbeddingSpace(dim=4)
    with pytest.raises(ValueError, match="dim"):
        space.set("bad", [1.0, 2.0])   # only 2 elements


def test_remove():
    space = EmbeddingSpace(dim=2)
    space.set("a", [1.0, 0.0])
    space.remove("a")
    assert space.get("a") is None


def test_require_missing_raises():
    space = EmbeddingSpace(dim=2)
    with pytest.raises(KeyError):
        space.require("missing")


def test_count():
    space = EmbeddingSpace(dim=2)
    space.set("x", [1.0, 0.0])
    space.set("y", [0.0, 1.0])
    assert space.count == 2


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def test_cosine_identical():
    space = EmbeddingSpace(dim=3)
    v = [1.0, 2.0, 3.0]
    space.set("a", v)
    space.set("b", v)
    score = space.cosine_similarity(space.require("a"), space.require("b"))
    assert abs(score - 1.0) < 1e-6


def test_cosine_orthogonal():
    space = EmbeddingSpace(dim=2)
    space.set("x", [1.0, 0.0])
    space.set("y", [0.0, 1.0])
    score = space.cosine_similarity(space.require("x"), space.require("y"))
    assert abs(score) < 1e-6


def test_cosine_opposite():
    space = EmbeddingSpace(dim=2)
    space.set("pos", [1.0, 0.0])
    space.set("neg", [-1.0, 0.0])
    score = space.cosine_similarity(space.require("pos"), space.require("neg"))
    assert abs(score + 1.0) < 1e-6


def test_cosine_zero_vector():
    space = EmbeddingSpace(dim=2)
    space.set("zero", [0.0, 0.0])
    space.set("one", [1.0, 0.0])
    score = space.cosine_similarity(space.require("zero"), space.require("one"))
    assert score == 0.0


# ---------------------------------------------------------------------------
# most_similar / KNN
# ---------------------------------------------------------------------------

def test_most_similar_returns_top_k():
    space = EmbeddingSpace(dim=2)
    space.set("a", [1.0, 0.0])
    space.set("b", [0.9, 0.1])
    space.set("c", [0.0, 1.0])
    space.set("d", [-1.0, 0.0])
    results = space.most_similar([1.0, 0.0], top_k=2)
    assert len(results) == 2
    top_keys = [k for k, _ in results]
    assert "a" in top_keys  # most similar to query [1.0, 0.0]


def test_knn_alias():
    space = EmbeddingSpace(dim=2)
    space.set("x", [1.0, 0.0])
    space.set("y", [0.0, 1.0])
    results = space.knn([1.0, 0.0], k=1)
    assert results[0][0] == "x"


def test_most_similar_dot_metric():
    space = EmbeddingSpace(dim=2)
    space.set("big", [10.0, 0.0])
    space.set("small", [0.1, 0.0])
    results = space.most_similar([1.0, 0.0], top_k=1, metric="dot")
    assert results[0][0] == "big"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_compose_mean():
    space = EmbeddingSpace(dim=2)
    space.set("a", [1.0, 0.0])
    space.set("b", [0.0, 1.0])
    mean = space.compose_mean(["a", "b"])
    assert mean._data == [0.5, 0.5]


def test_compose_sum():
    space = EmbeddingSpace(dim=2)
    space.set("a", [1.0, 2.0])
    space.set("b", [3.0, 4.0])
    s = space.compose_sum(["a", "b"])
    assert s._data == [4.0, 6.0]


def test_compose_mean_empty():
    space = EmbeddingSpace(dim=3)
    mean = space.compose_mean([])
    assert mean._data == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_from_dict():
    space = EmbeddingSpace(dim=2, name="test-emb")
    space.set("a", [1.0, 2.0])
    space.set("b", [3.0, 4.0])
    d = space.to_dict()
    space2 = EmbeddingSpace.from_dict(d)
    assert space2.dim == 2
    assert space2.name == "test-emb"
    assert space2.require("a")._data == [1.0, 2.0]
    assert space2.require("b")._data == [3.0, 4.0]


def test_to_from_json():
    space = EmbeddingSpace(dim=3, name="json-test")
    space.set("hello", [0.1, 0.2, 0.3])
    json_str = space.to_json()
    space2 = EmbeddingSpace.from_json(json_str)
    assert abs(space2.require("hello")._data[0] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# 32-bit memory page serialisation
# ---------------------------------------------------------------------------

def test_memory_page_roundtrip():
    space = EmbeddingSpace(dim=4, name="mem-test")
    space.set("alpha", [1.0, 2.0, 3.0, 4.0])
    space.set("beta",  [5.0, 6.0, 7.0, 8.0])
    page   = space.to_memory_page()
    space2 = EmbeddingSpace.from_memory_page(page, name="mem-test")
    assert space2.count == 2
    a = space2.require("alpha")
    assert len(a._data) == 4
    # Float32 precision
    assert abs(a._data[0] - 1.0) < 1e-4
    assert abs(a._data[3] - 4.0) < 1e-4


def test_memory_page_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        EmbeddingSpace.from_memory_page(b"\x00" * 64)


def test_memory_page_too_short():
    with pytest.raises(ValueError, match="too short"):
        EmbeddingSpace.from_memory_page(b"\x01\x02")
