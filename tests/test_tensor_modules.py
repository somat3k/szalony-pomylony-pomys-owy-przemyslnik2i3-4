"""Tests for the new tensor subsystem modules:
tensor/helpers, tensor/ops, tensor/matrix, tensor/transforms,
tensor/hyperparams, tensor/batch, tensor/graph_node.
"""

from __future__ import annotations

import math
import pytest

from hololang.tensor.tensor import Tensor
from hololang.tensor.helpers import strides, flat_index, total_elements
from hololang.tensor import (
    GraphNode, ComputationGraph, SpreadTensor, BatchContainer,
    MatrixEngine, ParameterMultiplier,
    TransformChain, ParallelTransformGroup,
    HyperParameter, HyperParamSpace,
    relu, leaky_relu, sigmoid, tanh_act, elu, softmax,
    layer_norm, min_max_scale, dropout, clip, scale, offset,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_total_elements(self):
        assert total_elements([3, 4]) == 12
        assert total_elements([2, 3, 4]) == 24
        assert total_elements([5]) == 5

    def test_strides_2d(self):
        assert strides([4, 3]) == [3, 1]

    def test_strides_3d(self):
        assert strides([2, 3, 4]) == [12, 4, 1]

    def test_flat_index(self):
        s = strides([4, 3])
        assert flat_index((0, 0), s) == 0
        assert flat_index((1, 0), s) == 3
        assert flat_index((2, 2), s) == 8


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

class TestOps:
    def _t(self, data):
        return Tensor([len(data)], data)

    def test_relu(self):
        t = self._t([-2.0, 0.0, 3.0])
        r = relu(t)
        assert r._data == [0.0, 0.0, 3.0]

    def test_leaky_relu(self):
        t = self._t([-1.0, 2.0])
        r = leaky_relu(t, alpha=0.1)
        assert abs(r._data[0] - (-0.1)) < 1e-9
        assert r._data[1] == 2.0

    def test_sigmoid_range(self):
        t = self._t([-10.0, 0.0, 10.0])
        r = sigmoid(t)
        assert r._data[0] < 0.01
        assert abs(r._data[1] - 0.5) < 1e-9
        assert r._data[2] > 0.99

    def test_tanh(self):
        t = self._t([0.0])
        assert abs(tanh_act(t)._data[0]) < 1e-9

    def test_elu_positive(self):
        t = self._t([1.0, -1.0])
        r = elu(t)
        assert r._data[0] == 1.0
        assert r._data[1] == pytest.approx(math.exp(-1.0) - 1.0, abs=1e-9)

    def test_softmax_sums_to_one(self):
        t = self._t([1.0, 2.0, 3.0])
        r = softmax(t)
        assert abs(sum(r._data) - 1.0) < 1e-9

    def test_layer_norm_zero_mean(self):
        t = self._t([1.0, 2.0, 3.0, 4.0])
        r = layer_norm(t)
        mean = sum(r._data) / len(r._data)
        assert abs(mean) < 1e-6

    def test_min_max_scale(self):
        t = self._t([0.0, 5.0, 10.0])
        r = min_max_scale(t, 0.0, 1.0)
        assert r._data == pytest.approx([0.0, 0.5, 1.0])

    def test_dropout_zero_rate(self):
        t = self._t([1.0, 2.0, 3.0])
        r = dropout(t, rate=0.0)
        assert r._data == t._data

    def test_dropout_invalid_rate_one(self):
        t = self._t([1.0, 2.0])
        with pytest.raises(ValueError, match="rate must be in"):
            dropout(t, rate=1.0)

    def test_dropout_invalid_rate_above_one(self):
        t = self._t([1.0, 2.0])
        with pytest.raises(ValueError, match="rate must be in"):
            dropout(t, rate=1.5)

    def test_dropout_invalid_rate_negative(self):
        t = self._t([1.0, 2.0])
        with pytest.raises(ValueError, match="rate must be in"):
            dropout(t, rate=-0.1)

    def test_sigmoid_large_negative(self):
        """Stable sigmoid must not raise OverflowError for large negative input."""
        t = self._t([-1000.0, 0.0, 1000.0])
        r = sigmoid(t)
        assert r._data[0] < 1e-6
        assert abs(r._data[1] - 0.5) < 1e-9
        assert r._data[2] > 1.0 - 1e-6

    def test_clip(self):
        t = self._t([-5.0, 0.5, 10.0])
        r = clip(t, 0.0, 1.0)
        assert r._data == [0.0, 0.5, 1.0]

    def test_scale(self):
        t = self._t([1.0, 2.0])
        assert scale(t, 3.0)._data == [3.0, 6.0]

    def test_offset(self):
        t = self._t([1.0, 2.0])
        assert offset(t, 10.0)._data == [11.0, 12.0]


# ---------------------------------------------------------------------------
# matrix engine
# ---------------------------------------------------------------------------

class TestMatrixEngine:
    def test_matmul_identity(self):
        engine = MatrixEngine(workers=2, block_rows=2)
        a = Tensor.eye(3)
        b = Tensor.ones(3, 3)
        c = engine.matmul(a, b)
        assert c._data == b._data
        engine.close()

    def test_matmul_shape_mismatch(self):
        engine = MatrixEngine(workers=1)
        a = Tensor([2, 3])
        b = Tensor([4, 2])
        with pytest.raises(ValueError):
            engine.matmul(a, b)
        engine.close()

    def test_batch_matmul(self):
        engine = MatrixEngine(workers=2)
        pairs = [(Tensor.eye(2), Tensor.eye(2)) for _ in range(4)]
        results = engine.batch_matmul(pairs)
        assert len(results) == 4
        for r in results:
            assert r._data == Tensor.eye(2)._data
        engine.close()

    def test_matmul_correctness(self):
        engine = MatrixEngine(workers=1)
        a = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
        b = Tensor([2, 2], [5.0, 6.0, 7.0, 8.0])
        c = engine.matmul(a, b)
        expected = a.matmul(b)
        assert c._data == pytest.approx(expected._data)
        engine.close()


# ---------------------------------------------------------------------------
# ParameterMultiplier
# ---------------------------------------------------------------------------

class TestParameterMultiplier:
    def test_constant(self):
        pm = ParameterMultiplier(base_value=0.01, policy="constant")
        assert pm.value() == pytest.approx(0.01)
        assert pm.step() == pytest.approx(0.01)

    def test_warmup(self):
        pm = ParameterMultiplier(base_value=1.0, policy="warmup", warmup_steps=10)
        assert pm.value(0) == pytest.approx(0.0)
        assert pm.value(5) == pytest.approx(0.5)
        assert pm.value(10) == pytest.approx(1.0)
        assert pm.value(20) == pytest.approx(1.0)

    def test_decay(self):
        pm = ParameterMultiplier(base_value=1.0, policy="decay",
                                  decay_rate=0.5, min_value=0.0)
        assert pm.value(0) == pytest.approx(1.0)
        assert pm.value(2) == pytest.approx(0.25)

    def test_step_policy(self):
        pm = ParameterMultiplier(base_value=1.0, policy="step",
                                  decay_rate=0.5, step_size=2)
        assert pm.value(0) == pytest.approx(1.0)
        assert pm.value(2) == pytest.approx(0.5)
        assert pm.value(4) == pytest.approx(0.25)

    def test_cyclic(self):
        pm = ParameterMultiplier(base_value=1.0, policy="cyclic",
                                  min_value=0.0, cycle_steps=4)
        v0 = pm.value(0)
        v2 = pm.value(2)
        assert v0 > v2  # at start (t=0) cosine is 1; at half-period it's 0

    def test_apply(self):
        pm = ParameterMultiplier(base_value=2.0, policy="constant")
        t = Tensor([3], [1.0, 2.0, 3.0])
        r = pm.apply(t)
        assert r._data == pytest.approx([2.0, 4.0, 6.0])

    def test_invalid_policy(self):
        with pytest.raises(ValueError):
            ParameterMultiplier(policy="bogus")


# ---------------------------------------------------------------------------
# TransformChain
# ---------------------------------------------------------------------------

class TestTransformChain:
    def test_sequential(self):
        chain = TransformChain("test").add(relu).add(lambda t: scale(t, 2.0))
        t = Tensor([3], [-1.0, 0.0, 3.0])
        r = chain(t)
        assert r._data == pytest.approx([0.0, 0.0, 6.0])

    def test_trace(self):
        chain = TransformChain().add(relu, "relu").add(lambda t: scale(t, 2.0), "x2")
        t = Tensor([2], [-1.0, 2.0])
        steps = chain.trace(t)
        assert len(steps) == 2
        names = [n for n, _ in steps]
        assert "relu" in names and "x2" in names

    def test_then(self):
        c1 = TransformChain("a").add(relu)
        c2 = TransformChain("b").add(lambda t: scale(t, 3.0))
        combined = c1.then(c2)
        t = Tensor([2], [-1.0, 2.0])
        r = combined(t)
        assert r._data == pytest.approx([0.0, 6.0])

    def test_len_repr(self):
        chain = TransformChain("x").add(relu).add(sigmoid)
        assert len(chain) == 2
        assert "x" in repr(chain)

    def test_remove_only_first_match(self):
        """remove() should delete only the first occurrence of a label."""
        chain = (TransformChain()
                 .add(relu, "step")
                 .add(lambda t: scale(t, 2.0), "step")
                 .add(lambda t: offset(t, 1.0), "other"))
        chain.remove("step")
        names = [n for n, _ in chain]
        assert names.count("step") == 1  # second "step" remains
        assert "other" in names

    def test_parallel_group(self):
        group = (ParallelTransformGroup()
                 .add(relu, "relu")
                 .add(lambda t: scale(t, 2.0), "double"))
        t = Tensor([3], [-1.0, 0.0, 2.0])
        results = group(t)
        assert set(results.keys()) == {"relu", "double"}
        assert results["relu"]._data == pytest.approx([0.0, 0.0, 2.0])
        assert results["double"]._data == pytest.approx([-2.0, 0.0, 4.0])


# ---------------------------------------------------------------------------
# HyperParamSpace
# ---------------------------------------------------------------------------

class TestHyperParamSpace:
    def _space(self):
        return (
            HyperParamSpace("test")
            .add("lr", 0.01, description="learning rate",
                  min_value=0.0, max_value=1.0)
            .add("momentum", 0.9)
            .add("batch_size", 32.0, dtype="int")
        )

    def test_access(self):
        space = self._space()
        assert space.value("lr") == pytest.approx(0.01)
        assert space["momentum"].value == pytest.approx(0.9)

    def test_contains(self):
        space = self._space()
        assert "lr" in space
        assert "unknown" not in space

    def test_missing_key(self):
        space = self._space()
        with pytest.raises(KeyError):
            space.get("not_exist")

    def test_effective_values(self):
        space = self._space()
        evs = space.effective_values()
        assert set(evs.keys()) == {"lr", "momentum", "batch_size"}

    def test_with_multiplier(self):
        pm = ParameterMultiplier(base_value=1.0, policy="constant")
        space = HyperParamSpace("x")
        space.add("alpha", 0.5, multiplier=pm)
        assert space.value("alpha") == pytest.approx(0.5)

    def test_serialization(self):
        space = self._space()
        d = space.to_dict()
        restored = HyperParamSpace.from_dict(d, "test2")
        assert restored.value("lr") == pytest.approx(0.01)
        assert restored.value("momentum") == pytest.approx(0.9)

    def test_step_all(self):
        space = self._space()
        evs = space.step_all()
        assert "lr" in evs


# ---------------------------------------------------------------------------
# SpreadTensor / BatchContainer
# ---------------------------------------------------------------------------

class TestBatch:
    def test_spread_gather_scatter(self):
        parent = Tensor([6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        shard = SpreadTensor(parent, offset=2, length=3, shard_index=1)
        gathered = shard.gather()
        assert gathered._data == [3.0, 4.0, 5.0]

        # scatter new values back
        new_t = Tensor([3], [10.0, 20.0, 30.0])
        shard.scatter(new_t)
        assert parent._data[2:5] == [10.0, 20.0, 30.0]

    def test_spread_scatter_wrong_size(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        shard = SpreadTensor(parent, 0, 4)
        with pytest.raises(ValueError):
            shard.scatter(Tensor([2], [1.0, 2.0]))

    def test_batch_from_tensor(self):
        parent = Tensor([9], list(range(1, 10, 1)))
        batch = BatchContainer.from_tensor(parent, n_shards=3)
        assert len(batch) == 3
        assert batch[0].length == 3
        assert batch[2].length == 3

    def test_batch_uneven(self):
        parent = Tensor([10], list(float(i) for i in range(10)))
        batch = BatchContainer.from_tensor(parent, n_shards=3)
        lengths = [s.length for s in batch.shards]
        assert sum(lengths) == 10
        assert len(batch) == 3

    def test_batch_apply_sequential(self):
        parent = Tensor([6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        batch = BatchContainer.from_tensor(parent, n_shards=2)
        results = batch.apply(lambda t: scale(t, 2.0), workers=1)
        assert len(results) == 2
        # parent should be mutated
        assert parent._data == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

    def test_batch_apply_parallel(self):
        parent = Tensor([6], [1.0] * 6)
        batch = BatchContainer.from_tensor(parent, n_shards=3)
        results = batch.apply(lambda t: scale(t, 3.0), workers=3)
        assert len(results) == 3

    def test_gather_all(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        batch = BatchContainer.from_tensor(parent, n_shards=2)
        all_t = batch.gather_all()
        assert all_t._data == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_reduce(self):
        parent = Tensor([6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        batch = BatchContainer.from_tensor(parent, n_shards=3)
        total = batch.reduce(lambda acc, t: acc + t)
        assert total is not None

    def test_batch_invalid_shards(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            BatchContainer.from_tensor(parent, n_shards=0)

    def test_spread_tensor_negative_offset(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="offset must be >= 0"):
            SpreadTensor(parent, offset=-1, length=2)

    def test_spread_tensor_negative_length(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="length must be >= 0"):
            SpreadTensor(parent, offset=0, length=-1)

    def test_spread_tensor_out_of_bounds(self):
        parent = Tensor([4], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="exceeds parent size"):
            SpreadTensor(parent, offset=3, length=3)


# ---------------------------------------------------------------------------
# GraphNode (standalone)
# ---------------------------------------------------------------------------

class TestGraphNode:
    def test_standalone_import(self):
        from hololang.tensor.graph_node import GraphNode as GN
        t = Tensor([2], [1.0, 2.0])
        node = GN(op="constant", inputs=[], fn=lambda: t, name="x")
        assert node.op == "constant"
        assert node.name == "x"
        assert node.output is None

    def test_graph_node_repr(self):
        from hololang.tensor.graph_node import GraphNode as GN
        t = Tensor([1], [0.0])
        node = GN("relu", [], lambda: t)
        assert "relu" in repr(node)
