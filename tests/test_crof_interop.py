"""Tests for CROF Interop Plane (Plane F) — LayerZero + gRPC + MQTT adapters."""

import pytest
from hololang.crof.envelope import Envelope, OperationType
from hololang.crof.interop import (
    GrpcAdapter, MqttAdapter,
    LayerZeroAdapter, LzPacket, DvnVerification,
    DispatchResult,
)


def _env(**kw) -> Envelope:
    return Envelope.build(
        actor_id="actor-1",
        domain_id="interop-domain",
        **kw,
    )


# ---------------------------------------------------------------------------
# DispatchResult
# ---------------------------------------------------------------------------

def test_dispatch_result_ok():
    dr = DispatchResult(status="dispatched")
    assert dr.ok


def test_dispatch_result_failed():
    dr = DispatchResult(status="failed", error="network error")
    assert not dr.ok


# ---------------------------------------------------------------------------
# GrpcAdapter
# ---------------------------------------------------------------------------

def test_grpc_dispatch_no_hook():
    adapter = GrpcAdapter(endpoint="localhost:50051")
    env = _env()
    result = adapter.dispatch(env)
    assert result.adapter_type == "grpc"
    assert result.status == "dispatched"


def test_grpc_dispatch_with_hook():
    received: list[tuple] = []

    def hook(method: str, payload: bytes) -> bytes:
        received.append((method, payload))
        return b""

    adapter = GrpcAdapter(endpoint="localhost:50051")
    adapter.register_call_hook(hook)
    env = _env()
    result = adapter.dispatch(env)
    assert result.ok
    assert len(received) == 1
    assert "DispatchInterdomain" in received[0][0]


def test_grpc_dispatch_hook_raises():
    def bad_hook(method, payload):
        raise ConnectionRefusedError("refused")

    adapter = GrpcAdapter(endpoint="localhost:50051")
    adapter.register_call_hook(bad_hook)
    env = _env()
    result = adapter.dispatch(env)
    assert not result.ok
    assert "refused" in result.error


def test_grpc_query_remote_state():
    adapter = GrpcAdapter(endpoint="localhost:50051")
    state = adapter.query_remote_state({"key": "val"})
    assert isinstance(state, dict)


# ---------------------------------------------------------------------------
# MqttAdapter
# ---------------------------------------------------------------------------

def test_mqtt_dispatch_no_hook():
    adapter = MqttAdapter(broker="localhost:1883")
    env = _env()
    result = adapter.dispatch(env)
    assert result.adapter_type == "mqtt"
    assert result.ok


def test_mqtt_dispatch_with_hook():
    published: list[tuple] = []

    def hook(topic: str, payload: bytes, qos: int) -> None:
        published.append((topic, payload, qos))

    adapter = MqttAdapter(broker="localhost:1883", topic_prefix="crof/test", qos=1)
    adapter.register_publish_hook(hook)
    env = _env(operation_type=OperationType.EVENT.value)
    result = adapter.dispatch(env)
    assert result.ok
    assert len(published) == 1
    topic, payload, qos = published[0]
    assert "crof/test" in topic
    assert qos == 1


def test_mqtt_topic_includes_domain():
    published: list[str] = []

    def hook(topic, payload, qos):
        published.append(topic)

    adapter = MqttAdapter(broker="b:1883", topic_prefix="prefix")
    adapter.register_publish_hook(hook)
    adapter.dispatch(_env())
    assert "interop-domain" in published[0]


# ---------------------------------------------------------------------------
# LayerZeroAdapter — packet construction
# ---------------------------------------------------------------------------

def test_lz_packet_encode():
    pkt = LzPacket(src_eid=30101, dst_eid=30110,
                   sender="0xAAA", receiver="0xBBB",
                   nonce=1, message=b"hello")
    encoded = pkt.encode()
    assert b"hello" in encoded


def test_lz_packet_keccak_guid():
    pkt1 = LzPacket(src_eid=1, dst_eid=2, sender="A", receiver="B",
                    nonce=1, message=b"msg")
    pkt2 = LzPacket(src_eid=1, dst_eid=2, sender="A", receiver="B",
                    nonce=2, message=b"msg")
    # Different nonce → different GUID
    assert pkt1.keccak_guid != pkt2.keccak_guid


def test_lz_dispatch_basic():
    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    env = _env()
    result = adapter.dispatch(env)
    assert result.adapter_type == "layerzero"
    assert result.ok
    assert len(adapter.sent_packets) == 1


def test_lz_nonce_increments():
    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    adapter.dispatch(_env())
    adapter.dispatch(_env())
    assert adapter.sent_packets[0].nonce == 1
    assert adapter.sent_packets[1].nonce == 2


def test_lz_dispatch_with_send_hook():
    hooked: list[LzPacket] = []

    def hook(pkt: LzPacket) -> None:
        hooked.append(pkt)

    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110, send_hook=hook)
    env = _env(payload=b"cross-chain")
    result = adapter.dispatch(env)
    assert result.ok
    assert len(hooked) == 1
    # Payload is JSON-encoded (hex-encoded bytes inside JSON); just verify non-empty
    assert len(hooked[0].message) > 0


def test_lz_send_hook_raises_gives_error():
    def bad_hook(pkt):
        raise OSError("endpoint down")

    adapter = LayerZeroAdapter(send_hook=bad_hook)
    result = adapter.dispatch(_env())
    assert not result.ok
    assert "endpoint down" in result.error


# ---------------------------------------------------------------------------
# LayerZeroAdapter — DVN simulation
# ---------------------------------------------------------------------------

def test_lz_dvn_simulate():
    adapter = LayerZeroAdapter(dvn_count=3)
    pkt = LzPacket(src_eid=30101, dst_eid=30110, nonce=1, message=b"test")
    pkt.guid = pkt.keccak_guid
    receipts = adapter.simulate_dvn_receive(pkt)
    assert len(receipts) == 3
    for r in receipts:
        assert isinstance(r, DvnVerification)
        assert r.verified
        assert r.packet_guid == pkt.guid


def test_lz_dvn_receipts_recorded():
    adapter = LayerZeroAdapter(dvn_count=2)
    pkt = LzPacket(nonce=1, message=b"x")
    pkt.guid = pkt.keccak_guid
    adapter.simulate_dvn_receive(pkt)
    assert len(adapter.dvn_receipts) == 2


# ---------------------------------------------------------------------------
# LayerZeroAdapter — query_remote_state
# ---------------------------------------------------------------------------

def test_lz_query_remote_state():
    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    adapter.dispatch(_env())
    state = adapter.query_remote_state({"block": 999})
    assert state["packets_sent"] == 1
    assert state["src_eid"] == 30101


# ---------------------------------------------------------------------------
# Metadata in dispatch results
# ---------------------------------------------------------------------------

def test_lz_result_metadata_contains_guid():
    adapter = LayerZeroAdapter()
    result = adapter.dispatch(_env())
    assert "lz_guid" in result.metadata
    assert "nonce" in result.metadata
