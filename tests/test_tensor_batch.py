"""Tests for SpreadTensor and BatchContainer (hololang/tensor/batch.py)."""

import json
import struct
import pytest

from hololang.tensor.tensor import Tensor
from hololang.tensor.batch import SpreadTensor, BatchContainer


# ===========================================================================
# SpreadTensor — basic
# ===========================================================================

def _t(n: int, d: int = 1, val: float = 1.0) -> Tensor:
    """Create a simple 2-D tensor of shape (n, d) filled with val."""
    if d == 1:
        return Tensor([n], [val] * n)
    return Tensor([n, d], [val] * (n * d))


def test_spread_n_batches_exact():
    st = SpreadTensor(_t(6), batch_size=2)
    assert st.n_batches == 3


def test_spread_n_batches_with_remainder():
    st = SpreadTensor(_t(7), batch_size=3)
    assert st.n_batches == 3  # ceil(7/3) = 3


def test_spread_n_batches_single():
    st = SpreadTensor(_t(4), batch_size=10)
    assert st.n_batches == 1


def test_spread_returns_correct_count():
    batches = SpreadTensor(_t(10), batch_size=3).spread()
    assert len(batches) == 4   # 3+3+3+1


def test_spread_batch_sizes():
    batches = SpreadTensor(_t(7), batch_size=3).spread()
    assert batches[0].dims[0] == 3
    assert batches[1].dims[0] == 3
    assert batches[2].dims[0] == 1


def test_spread_last_batch_has_remainder():
    st = SpreadTensor(_t(7), batch_size=3)
    assert st.remainder == 1


def test_spread_remainder_full_batch():
    st = SpreadTensor(_t(6), batch_size=3)
    assert st.remainder == 3   # last batch is exactly full


def test_spread_empty_tensor():
    t  = Tensor([0], [])
    st = SpreadTensor(t, batch_size=4)
    assert st.n_batches == 0
    assert st.spread() == []


def test_spread_single_sample():
    batches = SpreadTensor(_t(1), batch_size=32).spread()
    assert len(batches) == 1
    assert batches[0].dims[0] == 1


def test_spread_data_integrity():
    data = list(range(9))
    t    = Tensor([9], data)
    batches = SpreadTensor(t, batch_size=4).spread()
    reconstructed = []
    for b in batches:
        reconstructed.extend(b._data)
    assert reconstructed == [float(x) for x in data]


def test_spread_2d_tensor_dims():
    t  = Tensor([6, 4], [1.0] * 24)
    st = SpreadTensor(t, batch_size=2)
    batches = st.spread()
    assert len(batches) == 3
    for b in batches:
        assert b.dims == [2, 4]


def test_spread_2d_data_integrity():
    data = list(range(12))
    t    = Tensor([4, 3], [float(x) for x in data])
    batches = SpreadTensor(t, batch_size=2).spread()
    flat = []
    for b in batches:
        flat.extend(b._data)
    assert flat == [float(x) for x in data]


def test_spread_invalid_batch_size():
    with pytest.raises(ValueError):
        SpreadTensor(_t(4), batch_size=0)


def test_spread_scalar_tensor_raises():
    # A 0-dim tensor (dims=[]) is not constructable with Tensor([],[]) due to
    # data-length checks; instead verify that SpreadTensor rejects a valid
    # scalar tensor with dims=[].
    t = Tensor([], [1.0])   # scalar: dims=[], data=[1.0] (total=1 is valid)
    with pytest.raises(ValueError):
        SpreadTensor(t, batch_size=1)


def test_spread_repr():
    st = SpreadTensor(_t(10), batch_size=3)
    assert "SpreadTensor" in repr(st)
    assert "10" in repr(st)


def test_spread_name_propagated():
    t = Tensor([4], name="my_tensor")
    st = SpreadTensor(t, batch_size=2)
    batches = st.spread()
    assert "my_tensor" in batches[0].name or "spread" in batches[0].name


def test_spread_n_samples():
    st = SpreadTensor(_t(15), batch_size=4)
    assert st.n_samples == 15


# ===========================================================================
# BatchContainer — from_spread factory
# ===========================================================================

def test_batch_container_from_spread():
    t  = _t(10, d=4)
    bc = BatchContainer.from_spread(t, batch_size=3, model_target="scorer")
    assert bc.model_target == "scorer"
    assert bc.n_batches == 4
    assert bc.total_items == 10


def test_batch_container_total_items():
    bc = BatchContainer.from_spread(_t(7), batch_size=3)
    assert bc.total_items == 7


def test_batch_container_batch_size_property():
    bc = BatchContainer.from_spread(_t(8), batch_size=4)
    assert bc.batch_size == 4


def test_batch_container_feature_dims_1d():
    bc = BatchContainer.from_spread(_t(6), batch_size=3)
    assert bc.feature_dims == []   # 1-D source → no feature dims


def test_batch_container_feature_dims_2d():
    bc = BatchContainer.from_spread(_t(6, d=8), batch_size=3)
    assert bc.feature_dims == [8]


def test_batch_container_provenance():
    bc = BatchContainer.from_spread(_t(4), provenance=["env-1", "env-2"])
    assert "env-1" in bc.provenance
    assert "env-2" in bc.provenance


def test_batch_container_metadata():
    bc = BatchContainer.from_spread(_t(4), metadata={"key": "val"})
    assert bc.metadata["key"] == "val"


# ===========================================================================
# BatchContainer — as_flat reassembly
# ===========================================================================

def test_batch_container_as_flat_1d():
    data = list(range(9))
    t    = Tensor([9], [float(x) for x in data])
    bc   = BatchContainer.from_spread(t, batch_size=4)
    flat = bc.as_flat()
    assert flat.dims == [9]
    assert list(flat._data) == [float(x) for x in data]


def test_batch_container_as_flat_2d():
    data = [float(x) for x in range(12)]
    t    = Tensor([4, 3], data)
    bc   = BatchContainer.from_spread(t, batch_size=2)
    flat = bc.as_flat()
    assert flat.dims == [4, 3]
    assert list(flat._data) == data


def test_batch_container_as_flat_empty():
    bc = BatchContainer()
    flat = bc.as_flat()
    assert flat.dims == [0]


# ===========================================================================
# BatchContainer — from_envelope_payload
# ===========================================================================

def _json_payload(dims, data):
    return json.dumps({"dims": dims, "data": data}).encode()


def test_from_envelope_json_basic():
    payload = _json_payload([6], list(range(6)))
    bc = BatchContainer.from_envelope_payload(payload, batch_size=3, envelope_id="e1")
    assert bc.n_batches == 2
    assert "e1" in bc.provenance


def test_from_envelope_json_2d():
    payload = _json_payload([4, 3], list(range(12)))
    bc = BatchContainer.from_envelope_payload(payload, batch_size=2)
    assert bc.n_batches == 2
    assert bc.feature_dims == [3]


def test_from_envelope_raw_float32():
    # Raw bytes instead of JSON
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    bc = BatchContainer.from_envelope_payload(raw, batch_size=2)
    assert bc.n_batches >= 1
    assert bc.total_items >= 1


def test_from_envelope_empty_payload():
    bc = BatchContainer.from_envelope_payload(b"", envelope_id="empty")
    assert bc.n_batches >= 1  # sentinel batch


def test_from_envelope_preserves_model_target():
    payload = _json_payload([4], [1.0, 2.0, 3.0, 4.0])
    bc = BatchContainer.from_envelope_payload(payload, model_target="my-model")
    assert bc.model_target == "my-model"


# ===========================================================================
# BatchContainer — to_dict serialisation
# ===========================================================================

def test_batch_container_to_dict():
    bc = BatchContainer.from_spread(_t(4), model_target="m", provenance=["e1"])
    d = bc.to_dict()
    assert d["model_target"] == "m"
    assert d["n_batches"] == bc.n_batches
    assert d["total_items"] == 4
    assert "e1" in d["provenance"]


def test_batch_container_repr():
    bc = BatchContainer.from_spread(_t(6), batch_size=3, model_target="foo")
    assert "BatchContainer" in repr(bc)
    assert "foo" in repr(bc)
