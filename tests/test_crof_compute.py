"""Tests for TFBackend, ModelDeployment, and TensorComputeNode (hololang/crof/compute.py)."""

import json
import pytest

from hololang.tensor.tensor import Tensor
from hololang.tensor.batch import BatchContainer
from hololang.tensor.embeddings import EmbeddingSpace
from hololang.crof.envelope import Envelope, OperationType
from hololang.crof.receiver import Receiver, ReceiverChain
from hololang.crof.compute import (
    TFBackend,
    MockTFBackend,
    TensorFlowBackend,
    ModelDeployment,
    ModelOutput,
    TensorComputeNode,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _env(payload: bytes = b"", **kw) -> Envelope:
    return Envelope.build(
        actor_id="actor-1",
        domain_id="compute-domain",
        payload=payload,
        **kw,
    )


def _json_payload(dims, data):
    return json.dumps({"dims": dims, "data": data}).encode()


def _t(n: int, d: int = 1, val: float = 1.0) -> Tensor:
    if d == 1:
        return Tensor([n], [val] * n)
    return Tensor([n, d], [val] * (n * d))


def _deployment(name="test-model", predict_fn=None) -> ModelDeployment:
    backend = MockTFBackend(predict_fn=predict_fn)
    return ModelDeployment(name=name, model_name=name, backend=backend)


# ===========================================================================
# MockTFBackend
# ===========================================================================

def test_mock_backend_is_available():
    b = MockTFBackend()
    assert b.is_available()


def test_mock_backend_name():
    b = MockTFBackend(backend_name="my-mock")
    assert b.name == "my-mock"


def test_mock_backend_identity_predict():
    b = MockTFBackend()
    t = _t(4, d=3)
    out = b.predict(t, "m")
    assert out.dims == t.dims
    assert list(out._data) == list(t._data)


def test_mock_backend_custom_fn():
    def double(batch, model_name):
        data = [x * 2 for x in batch._data]
        return Tensor(list(batch.dims), data)

    b = MockTFBackend(predict_fn=double)
    t = _t(3, val=1.5)
    out = b.predict(t, "m")
    assert all(abs(v - 3.0) < 1e-6 for v in out._data)


def test_mock_backend_call_count():
    b = MockTFBackend()
    b.predict(_t(2), "m")
    b.predict(_t(2), "m")
    assert b.call_count == 2


def test_mock_satisfies_protocol():
    b = MockTFBackend()
    assert isinstance(b, TFBackend)


def test_mock_backend_repr():
    b = MockTFBackend(backend_name="test")
    assert "mock" in repr(b) or "test" in repr(b)


# ===========================================================================
# TensorFlowBackend — availability and lazy import
# ===========================================================================

def test_tf_backend_name():
    b = TensorFlowBackend(backend_name="tf")
    assert b.name == "tf"


def test_tf_backend_is_available_false_without_tf():
    """When TF is not installed, is_available() must return False gracefully."""
    b = TensorFlowBackend()
    # We don't know if TF is installed; just assert it returns a bool
    result = b.is_available()
    assert isinstance(result, bool)


def test_tf_backend_predict_raises_without_tf_or_model():
    b = TensorFlowBackend()
    if not b.is_available():
        with pytest.raises(ImportError, match="tensorflow"):
            b.predict(_t(2), "nonexistent")
    else:
        # TF available but no model registered
        with pytest.raises((KeyError, Exception)):
            b.predict(_t(2), "nonexistent")


def test_tf_backend_register_model():
    b = TensorFlowBackend()
    # Register a mock object as the model
    b.register_model("my-model", object())
    # Model is stored; actual predict would need TF to work
    assert "my-model" in b._models


def test_tf_backend_repr():
    b = TensorFlowBackend(backend_name="tf-test")
    assert "tf-test" in repr(b)


# ===========================================================================
# ModelDeployment — basic
# ===========================================================================

def test_deployment_run_basic():
    dep = _deployment("scorer")
    bc  = BatchContainer.from_spread(_t(6), batch_size=3, model_target="scorer")
    out = dep.run(bc)
    assert isinstance(out, ModelOutput)
    assert out.ok
    assert out.n_batches == 2


def test_deployment_predictions_count_equals_batches():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(9), batch_size=3)
    out = dep.run(bc)
    assert len(out.predictions) == 3


def test_deployment_latencies_recorded():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(6), batch_size=3)
    out = dep.run(bc)
    assert len(out.latencies_ms) == 2
    assert all(ms >= 0 for ms in out.latencies_ms)


def test_deployment_provenance_preserved():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(4), provenance=["env-A", "env-B"])
    out = dep.run(bc)
    assert "env-A" in out.provenance
    assert "env-B" in out.provenance


def test_deployment_confidence_full_for_mock():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(4))
    out = dep.run(bc)
    assert out.confidence == 1.0


def test_deployment_confidence_fn():
    dep = _deployment()
    dep.confidence_fn = lambda t: 0.7
    bc  = BatchContainer.from_spread(_t(4), batch_size=2)
    out = dep.run(bc)
    assert abs(out.confidence - 0.7) < 1e-6


def test_deployment_run_count_increments():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(4))
    dep.run(bc)
    dep.run(bc)
    assert dep.run_count == 2


def test_deployment_avg_run_ms():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(4))
    dep.run(bc)
    assert dep.avg_run_ms >= 0


def test_deployment_as_flat_prediction():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(6), batch_size=3)
    out = dep.run(bc)
    flat = out.as_flat_prediction()
    assert flat.dims[0] == 6


def test_deployment_max_batch_size_resplits():
    dep = _deployment()
    dep.max_batch_size = 2
    bc  = BatchContainer.from_spread(_t(6), batch_size=6)   # one big batch
    out = dep.run(bc)
    # The big batch of 6 gets re-split into 3 sub-batches of 2
    assert len(out.predictions) == 3


def test_deployment_backend_error_recorded():
    def fail_fn(batch, model_name):
        raise RuntimeError("backend exploded")

    dep = ModelDeployment(
        name="broken",
        model_name="broken",
        backend=MockTFBackend(predict_fn=fail_fn),
    )
    bc  = BatchContainer.from_spread(_t(4), batch_size=2)
    out = dep.run(bc)
    assert not out.ok
    assert "backend exploded" in out.error


def test_deployment_metadata_contains_backend():
    dep = _deployment()
    bc  = BatchContainer.from_spread(_t(4))
    out = dep.run(bc)
    assert "backend" in out.metadata


def test_model_output_repr():
    out = ModelOutput(deployment_name="d", model_name="m")
    assert "d" in repr(out)


def test_model_output_total_latency():
    out = ModelOutput(latencies_ms=[10.0, 5.0])
    assert out.total_latency_ms == 15.0


def test_model_output_avg_latency():
    out = ModelOutput(latencies_ms=[10.0, 20.0])
    assert out.avg_latency_ms == 15.0


def test_deployment_repr():
    dep = _deployment("my-dep")
    assert "my-dep" in repr(dep)


# ===========================================================================
# TensorComputeNode — model routing
# ===========================================================================

def _model_node(model_name="scorer", predict_fn=None):
    """Create a TensorComputeNode wired to a model receiver."""
    chain = ReceiverChain("test-chain")
    chain.add(Receiver(
        name="model-recv",
        predicate=lambda env: env.module_id == model_name,
        target_layer="model",
        model_name=model_name,
        batch_size=4,
        priority=10,
    ))
    node = TensorComputeNode(module_id="tcn-test", chain=chain)
    dep  = ModelDeployment(
        name=model_name,
        model_name=model_name,
        backend=MockTFBackend(predict_fn=predict_fn),
    )
    node.register_deployment(dep)
    return node


def test_tcn_model_route_ok():
    node = _model_node("scorer")
    payload = _json_payload([8], [1.0] * 8)
    env = _env(payload=payload, module_id="scorer")
    result = node.execute(env)
    assert result.ok
    assert isinstance(result.output, ModelOutput)
    assert result.output.ok


def test_tcn_model_predictions_non_empty():
    node = _model_node("scorer")
    payload = _json_payload([4], [2.0] * 4)
    env = _env(payload=payload, module_id="scorer")
    result = node.execute(env)
    assert result.output.n_batches >= 1


def test_tcn_passthrough_when_no_match():
    chain = ReceiverChain()
    chain.add(Receiver(name="r", predicate=lambda e: False))
    node  = TensorComputeNode(chain=chain)
    result = node.execute(_env())
    assert result.ok
    assert result.output is None


def test_tcn_model_missing_deployment_sets_error():
    chain = ReceiverChain()
    chain.add(Receiver(
        name="r",
        predicate=lambda e: True,
        target_layer="model",
        model_name="missing-model",
    ))
    node   = TensorComputeNode(chain=chain)
    result = node.execute(_env(module_id="missing-model"))
    assert not result.ok
    assert "missing-model" in result.error


def test_tcn_invocation_count():
    node = _model_node()
    payload = _json_payload([4], [1.0] * 4)
    env = _env(payload=payload, module_id="scorer")
    node.execute(env)
    node.execute(env)
    assert node.invocations == 2


def test_tcn_confidence_from_model_output():
    node = _model_node()
    payload = _json_payload([4], [1.0] * 4)
    result = node.execute(_env(payload=payload, module_id="scorer"))
    assert 0.0 <= result.confidence <= 1.0


def test_tcn_provenance_contains_envelope_id():
    node = _model_node()
    payload = _json_payload([4], [1.0] * 4)
    env = _env(payload=payload, module_id="scorer")
    result = node.execute(env)
    assert env.envelope_id in result.provenance


def test_tcn_metadata_has_target_layer():
    node = _model_node()
    payload = _json_payload([4], [1.0] * 4)
    env = _env(payload=payload, module_id="scorer")
    result = node.execute(env)
    assert result.metadata["target_layer"] == "model"


# ===========================================================================
# TensorComputeNode — tensor routing
# ===========================================================================

def test_tcn_tensor_route():
    chain = ReceiverChain()
    chain.add(Receiver(
        name="tensor-recv",
        predicate=lambda e: e.operation_type == OperationType.TRANSFORM.value,
        target_layer="tensor",
        batch_size=3,
    ))
    node = TensorComputeNode(chain=chain)
    payload = _json_payload([9], list(range(9)))
    env = _env(payload=payload, operation_type=OperationType.TRANSFORM.value)
    result = node.execute(env)
    assert result.ok
    assert isinstance(result.output, BatchContainer)
    assert result.output.total_items == 9


def test_tcn_tensor_batch_size_respected():
    chain = ReceiverChain()
    chain.add(Receiver(
        name="r",
        predicate=lambda e: True,
        target_layer="tensor",
        batch_size=2,
    ))
    node = TensorComputeNode(chain=chain)
    payload = _json_payload([6], [1.0] * 6)
    result = node.execute(_env(payload=payload))
    bc = result.output
    assert bc.n_batches == 3


# ===========================================================================
# TensorComputeNode — embedding routing
# ===========================================================================

def test_tcn_embedding_route():
    space = EmbeddingSpace(dim=4, name="words")
    space.set("hello", [1.0, 0.0, 0.0, 0.0])
    space.set("world", [0.0, 1.0, 0.0, 0.0])

    chain = ReceiverChain()
    chain.add(Receiver(
        name="emb-recv",
        predicate=lambda e: e.operation_type == OperationType.QUERY.value,
        target_layer="embedding",
        model_name="words",
        batch_size=2,
    ))
    node = TensorComputeNode(chain=chain)
    node.register_embedding_space(space)

    payload = json.dumps({"vector": [1.0, 0.0, 0.0, 0.0]}).encode()
    result = node.execute(_env(payload=payload))
    assert result.ok
    results = result.output
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0][0] == "hello"   # most similar to [1,0,0,0]


def test_tcn_embedding_missing_space_sets_error():
    chain = ReceiverChain()
    chain.add(Receiver(
        name="e",
        predicate=lambda env: True,
        target_layer="embedding",
        model_name="nonexistent",
    ))
    node = TensorComputeNode(chain=chain)
    result = node.execute(_env())
    assert not result.ok
    assert "nonexistent" in result.error


# ===========================================================================
# TensorComputeNode — build factory
# ===========================================================================

def test_tcn_build_factory():
    node = TensorComputeNode.build(module_id="built-node")
    assert "built-node" in node.module_id
    assert node.chain is not None


def test_tcn_repr():
    node = TensorComputeNode(module_id="my-tcn")
    assert "my-tcn" in repr(node)


# ===========================================================================
# End-to-end: envelope → receiver → batch → model → flat prediction
# ===========================================================================

def test_e2e_pipeline():
    """Full pipeline: JSON envelope payload → spread → model → flat prediction."""
    N, D = 12, 8
    data = [float(i % 10) for i in range(N * D)]
    payload = _json_payload([N, D], data)

    # Custom backend: return sum of each row as a (B, 1) prediction
    def row_sum(batch: Tensor, model_name: str) -> Tensor:
        B = batch.dims[0]
        feat = batch.dims[1] if len(batch.dims) > 1 else 1
        sums = []
        for i in range(B):
            s = sum(batch._data[i * feat:(i + 1) * feat])
            sums.append(s)
        return Tensor([B, 1], sums)

    chain = ReceiverChain("e2e-chain")
    chain.add(Receiver(
        name="json-model",
        predicate=lambda env: env.payload_type == "application/json",
        target_layer="model",
        model_name="row-sum",
        batch_size=4,
        priority=10,
    ))

    node = TensorComputeNode(module_id="e2e-node", chain=chain)
    dep  = ModelDeployment(
        name="row-sum",
        model_name="row-sum",
        backend=MockTFBackend(predict_fn=row_sum),
    )
    node.register_deployment(dep)

    env = _env(payload=payload, payload_type="application/json")
    result = node.execute(env)

    assert result.ok
    out: ModelOutput = result.output
    assert out.n_batches == 3   # ceil(12 / 4)

    flat = out.as_flat_prediction()
    assert flat.dims[0] == N   # one prediction per sample
