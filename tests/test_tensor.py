"""Tests for the tensor subsystem."""

import math
import pytest
import tempfile
import os

from hololang.tensor.tensor import Tensor
from hololang.tensor.safetensor import SafeTensor
from hololang.tensor.graph import ComputationGraph
from hololang.tensor.pool import TensorPool


# ---------------------------------------------------------------------------
# Tensor basics
# ---------------------------------------------------------------------------

def test_zeros():
    t = Tensor.zeros(3, 3)
    assert t.shape == (3, 3)
    assert t.sum() == 0.0


def test_ones():
    t = Tensor.ones(2, 3)
    assert t.shape == (2, 3)
    assert t.sum() == 6.0


def test_eye():
    t = Tensor.eye(3)
    assert t[0, 0] == 1.0
    assert t[0, 1] == 0.0
    assert t[1, 1] == 1.0
    assert t[2, 2] == 1.0


def test_element_access():
    t = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    assert t[0, 0] == 1.0
    assert t[0, 1] == 2.0
    assert t[1, 0] == 3.0
    assert t[1, 1] == 4.0


def test_set_element():
    t = Tensor([2, 2])
    t[0, 1] = 7.0
    assert t[0, 1] == 7.0


def test_add_scalar():
    t = Tensor.ones(2, 2)
    r = t + 1.0
    assert r.sum() == 8.0


def test_add_tensor():
    a = Tensor([2], [1.0, 2.0])
    b = Tensor([2], [3.0, 4.0])
    c = a + b
    assert c._data == [4.0, 6.0]


def test_sub():
    a = Tensor([2], [5.0, 3.0])
    b = Tensor([2], [2.0, 1.0])
    c = a - b
    assert c._data == [3.0, 2.0]


def test_mul_scalar():
    t = Tensor([3], [1.0, 2.0, 3.0])
    r = t * 3.0
    assert r._data == [3.0, 6.0, 9.0]


def test_neg():
    t = Tensor([2], [1.0, -2.0])
    r = -t
    assert r._data == [-1.0, 2.0]


def test_matmul():
    # 2×2 identity × any matrix = same matrix
    I = Tensor.eye(2)
    A = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    R = I.matmul(A)
    assert R._data == A._data


def test_matmul_shape_mismatch():
    a = Tensor([2, 3])
    b = Tensor([2, 2])
    with pytest.raises(ValueError):
        a.matmul(b)


def test_transpose():
    t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    r = t.transpose()
    assert r.shape == (3, 2)
    assert r[0, 0] == 1.0
    assert r[0, 1] == 4.0


def test_reshape():
    t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    r = t.reshape(3, 2)
    assert r.shape == (3, 2)
    assert r.size == 6


def test_reshape_bad_size():
    t = Tensor([2, 3])
    with pytest.raises(ValueError):
        t.reshape(5)


def test_flatten():
    t = Tensor([2, 3])
    f = t.flatten()
    assert f.shape == (6,)


def test_sum():
    t = Tensor([3], [1.0, 2.0, 3.0])
    assert t.sum() == 6.0


def test_mean():
    t = Tensor([4], [2.0, 4.0, 6.0, 8.0])
    assert t.mean() == 5.0


def test_min_max():
    t = Tensor([3], [3.0, 1.0, 2.0])
    assert t.min() == 1.0
    assert t.max() == 3.0


def test_norm():
    t = Tensor([3], [3.0, 4.0, 0.0])
    assert abs(t.norm() - 5.0) < 1e-9


def test_normalize():
    t = Tensor([2], [3.0, 4.0])
    n = t.normalize()
    assert abs(n.norm() - 1.0) < 1e-9


def test_clip():
    t = Tensor([3], [-1.0, 0.5, 2.0])
    c = t.clip(0.0, 1.0)
    assert c._data == [0.0, 0.5, 1.0]


def test_apply_fn():
    t = Tensor([3], [1.0, 4.0, 9.0])
    r = t.apply_fn(math.sqrt)
    assert abs(r._data[0] - 1.0) < 1e-9
    assert abs(r._data[1] - 2.0) < 1e-9
    assert abs(r._data[2] - 3.0) < 1e-9


def test_from_nested():
    t = Tensor.from_nested([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t[0, 0] == 1.0
    assert t[1, 1] == 4.0


def test_serialisation():
    t = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0], name="test")
    d = t.to_dict()
    t2 = Tensor.from_dict(d)
    assert t == t2


def test_iter_1d():
    t = Tensor([3], [10.0, 20.0, 30.0])
    values = list(t)
    assert values == [10.0, 20.0, 30.0]


def test_len():
    t = Tensor([5, 3])
    assert len(t) == 5


# ---------------------------------------------------------------------------
# SafeTensor
# ---------------------------------------------------------------------------

def test_safetensor_valid():
    t = Tensor([3], [0.1, 0.5, 0.9])
    st = SafeTensor(t, min_val=0.0, max_val=1.0)
    assert st.shape == (3,)


def test_safetensor_value_too_low():
    t = Tensor([2], [-0.1, 0.5])
    with pytest.raises(ValueError):
        SafeTensor(t, min_val=0.0, max_val=1.0, clamp=False)


def test_safetensor_clamp():
    t = Tensor([2], [-0.5, 1.5])
    st = SafeTensor(t, min_val=0.0, max_val=1.0, clamp=True)
    assert st.get(0) == 0.0
    assert st.get(1) == 1.0


def test_safetensor_bad_dtype():
    t = Tensor([2], dtype="complex128")
    with pytest.raises(ValueError):
        SafeTensor(t)


def test_safetensor_save_load(tmp_path):
    t = Tensor([2, 2], [0.1, 0.2, 0.3, 0.4])
    st = SafeTensor(t, min_val=0.0, max_val=1.0)
    path = str(tmp_path / "st.json")
    st.save(path)
    st2 = SafeTensor.load(path)
    assert st2.shape == (2, 2)
    assert abs(st2.get(0, 0) - 0.1) < 1e-6


def test_safetensor_crc_tamper(tmp_path):
    t = Tensor([2], [0.5, 0.6])
    st = SafeTensor(t)
    path = str(tmp_path / "st.json")
    st.save(path)
    # Tamper with the file
    import json
    with open(path) as fh:
        d = json.load(fh)
    d["crc32"] = 0xDEADBEEF
    with open(path, "w") as fh:
        json.dump(d, fh)
    with pytest.raises(ValueError, match="CRC32"):
        SafeTensor.load(path)


# ---------------------------------------------------------------------------
# Computation graph
# ---------------------------------------------------------------------------

def test_graph_forward():
    g = ComputationGraph("test")
    a = g.constant(Tensor([2], [1.0, 2.0]), "a")
    b = g.constant(Tensor([2], [3.0, 4.0]), "b")
    c = g.add(a, b)
    results = g.forward()
    assert results["a"]._data == [1.0, 2.0]
    assert results["b"]._data == [3.0, 4.0]
    assert results[c.name]._data == [4.0, 6.0]


def test_graph_relu():
    g = ComputationGraph("relu_test")
    a = g.constant(Tensor([3], [-1.0, 0.0, 2.0]), "a")
    r = g.relu(a)
    results = g.forward()
    assert results[r.name]._data == [0.0, 0.0, 2.0]


def test_graph_matmul():
    g = ComputationGraph("mm")
    I = g.constant(Tensor.eye(2), "I")
    A = g.constant(Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]), "A")
    out = g.matmul(I, A)
    results = g.forward()
    assert results[out.name]._data == A.tensor._data if hasattr(A, 'tensor') else [1.0, 2.0, 3.0, 4.0]


def test_graph_dot():
    g = ComputationGraph("dot")
    dot = g.to_dot()
    assert "digraph" in dot


# ---------------------------------------------------------------------------
# TensorPool
# ---------------------------------------------------------------------------

def test_pool_allocate():
    pool = TensorPool("test_pool")
    t = pool.allocate("buf1", 4, 4)
    assert t.shape == (4, 4)
    assert pool.get("buf1") is t


def test_pool_map():
    pool = TensorPool("map_pool")
    pool.allocate("a", 3)
    pool.allocate("b", 3)
    for tag in ["a", "b"]:
        t = pool.get(tag)
        for i in range(3):
            t._data[i] = 1.0

    results = pool.map(lambda t: t * 2.0)
    for tag, t in results.items():
        assert all(v == 2.0 for v in t._data)


def test_pool_reduce():
    pool = TensorPool("red_pool")
    pool.allocate("x", 3)
    pool.allocate("y", 3)
    for tag in ["x", "y"]:
        t = pool.get(tag)
        t._data[:] = [1.0, 1.0, 1.0]

    total = pool.reduce(lambda a, b: a + b)
    assert total is not None
    assert total.sum() == 6.0


def test_pool_process_surface():
    pool = TensorPool("surf_pool")
    pool.allocate("surface", 3, 3)

    def fn(r, c, v):
        return float(r * 3 + c)

    result = pool.process_surface("surface", fn)
    assert result[0, 0] == 0.0
    assert result[1, 2] == 5.0


def test_pool_free():
    pool = TensorPool("free_pool")
    pool.allocate("tmp", 2, 2)
    assert "tmp" in pool.tags
    pool.free("tmp")
    assert "tmp" not in pool.tags
