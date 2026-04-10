"""Tests for CROF Interop Plane (Plane F) — LayerZero V2 OApp + gRPC + MQTT + Arbitrage."""

import struct

import pytest

from hololang.crof.envelope import Envelope, OperationType
from hololang.crof.interop import (
    # Core
    DispatchResult,
    # gRPC
    GrpcAdapter,
    # MQTT
    MqttAdapter, MqttWillMessage, MqttRetainedMessage,
    # LayerZero V2 structures
    LayerZeroAdapter,
    LzPacket, DvnVerification,
    ExecutorOptions, LzOptionType,
    MessagingFee, SendParam, Origin,
    LzReadRequest, LzReadResponse,
    # Arbitrage
    ArbitrageChannel, ArbitrageOpportunity,
)


def _env(**kw) -> Envelope:
    return Envelope.build(
        actor_id="actor-1",
        domain_id="interop-domain",
        **kw,
    )


# ===========================================================================
# DispatchResult
# ===========================================================================

def test_dispatch_result_ok():
    dr = DispatchResult(status="dispatched")
    assert dr.ok


def test_dispatch_result_failed():
    dr = DispatchResult(status="failed", error="network error")
    assert not dr.ok


def test_dispatch_result_error_without_failed_status():
    dr = DispatchResult(status="dispatched", error="something")
    assert not dr.ok


def test_dispatch_result_repr():
    dr = DispatchResult(adapter_type="grpc", status="dispatched")
    assert "grpc" in repr(dr)


# ===========================================================================
# GrpcAdapter — basic
# ===========================================================================

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
    assert len(received) >= 1
    assert "DispatchInterdomain" in received[0][0]


def test_grpc_dispatch_hook_raises():
    def bad_hook(method, payload):
        raise ConnectionRefusedError("refused")

    adapter = GrpcAdapter(endpoint="localhost:50051", max_retries=1)
    adapter.register_call_hook(bad_hook)
    env = _env()
    result = adapter.dispatch(env)
    assert not result.ok
    assert "refused" in result.error


def test_grpc_query_remote_state():
    adapter = GrpcAdapter(endpoint="localhost:50051")
    state = adapter.query_remote_state({"key": "val"})
    assert isinstance(state, dict)


# ===========================================================================
# GrpcAdapter — retry, interceptor, health-check
# ===========================================================================

def test_grpc_retry_exhausts_attempts():
    calls: list[int] = []

    def bad_hook(method, payload):
        calls.append(1)
        raise OSError("down")

    adapter = GrpcAdapter(endpoint="h:1", max_retries=3)
    adapter.register_call_hook(bad_hook)
    result = adapter.dispatch(_env())
    assert not result.ok
    assert len(calls) == 3


def test_grpc_retry_succeeds_on_second_attempt():
    calls: list[int] = []

    def flaky_hook(method, payload):
        calls.append(1)
        if len(calls) < 2:
            raise OSError("transient")
        return b""

    adapter = GrpcAdapter(endpoint="h:1", max_retries=3)
    adapter.register_call_hook(flaky_hook)
    result = adapter.dispatch(_env())
    assert result.ok
    assert len(calls) == 2


def test_grpc_interceptor_chain():
    log: list[str] = []

    def interceptor(method, payload):
        log.append("intercept")
        return payload + b"|intercepted"

    def hook(method, payload):
        log.append("hook")
        assert b"|intercepted" in payload
        return b""

    adapter = GrpcAdapter(endpoint="h:1")
    adapter.add_interceptor(interceptor)
    adapter.register_call_hook(hook)
    adapter.dispatch(_env())
    assert log == ["intercept", "hook"]


def test_grpc_interceptor_raises_gives_error():
    def bad_interceptor(method, payload):
        raise ValueError("intercept failure")

    adapter = GrpcAdapter(endpoint="h:1", max_retries=1)
    adapter.add_interceptor(bad_interceptor)
    result = adapter.dispatch(_env())
    assert not result.ok
    assert "intercept failure" in result.error


def test_grpc_health_check_no_hook():
    adapter = GrpcAdapter(endpoint="h:1")
    assert adapter.health_check() is True


def test_grpc_health_check_hook_ok():
    adapter = GrpcAdapter(endpoint="h:1")
    adapter.register_call_hook(lambda m, p: b"")
    assert adapter.health_check() is True


def test_grpc_health_check_hook_raises():
    adapter = GrpcAdapter(endpoint="h:1")
    adapter.register_call_hook(lambda m, p: (_ for _ in ()).throw(OSError("down")))
    assert adapter.health_check() is False


def test_grpc_call_log_records_attempts():
    adapter = GrpcAdapter(endpoint="h:1", max_retries=2)
    adapter.register_call_hook(lambda m, p: (_ for _ in ()).throw(OSError("x")))
    adapter.dispatch(_env())
    assert len(adapter.call_log) == 2


def test_grpc_metadata_stored():
    adapter = GrpcAdapter(endpoint="h:1", metadata={"auth": "Bearer tok"})
    assert adapter.metadata["auth"] == "Bearer tok"


# ===========================================================================
# MqttAdapter — basic
# ===========================================================================

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


# ===========================================================================
# MqttAdapter — subscriptions, retained messages, LWT, delivery counts
# ===========================================================================

def test_mqtt_subscribe_and_receive():
    received: list[tuple] = []
    adapter = MqttAdapter(broker="b:1")
    adapter.subscribe("crof/#", lambda t, p: received.append((t, p)))
    adapter.dispatch(_env())
    assert len(received) == 1
    assert "crof" in received[0][0]


def test_mqtt_topic_wildcard_plus():
    received: list[tuple] = []
    adapter = MqttAdapter(broker="b:1", topic_prefix="x")
    # "+/+" matches two-segment topics
    adapter.subscribe("+/+", lambda t, p: received.append(t))
    # topic is "x/{domain_id}/{op_type}" — three segments, should NOT match
    adapter.dispatch(_env())
    assert len(received) == 0


def test_mqtt_unsubscribe():
    received: list[tuple] = []
    adapter = MqttAdapter(broker="b:1")
    adapter.subscribe("crof/#", lambda t, p: received.append(t))
    adapter.unsubscribe("crof/#")
    adapter.dispatch(_env())
    assert len(received) == 0


def test_mqtt_retained_message():
    adapter = MqttAdapter(broker="b:1")
    adapter.dispatch(_env(), retain=True)
    assert len(adapter.retained) == 1


def test_mqtt_no_retain_by_default():
    adapter = MqttAdapter(broker="b:1")
    adapter.dispatch(_env())
    assert len(adapter.retained) == 0


def test_mqtt_delivery_count():
    adapter = MqttAdapter(broker="b:1", qos=2)
    adapter.dispatch(_env())
    adapter.dispatch(_env())
    assert adapter.delivery_counts[2] == 2


def test_mqtt_lwt_delivered_to_subscribers():
    received: list[bytes] = []
    will = MqttWillMessage(topic="crof/lwt", payload=b"offline", qos=1)
    adapter = MqttAdapter(broker="b:1", will=will)
    adapter.subscribe("crof/lwt", lambda t, p: received.append(p))
    adapter.trigger_lwt()
    assert received == [b"offline"]


def test_mqtt_lwt_retained():
    will = MqttWillMessage(topic="crof/lwt", payload=b"gone", qos=1, retain=True)
    adapter = MqttAdapter(broker="b:1", will=will)
    adapter.trigger_lwt()
    assert "crof/lwt" in adapter.retained


def test_mqtt_no_lwt_trigger_without_will():
    adapter = MqttAdapter(broker="b:1")
    adapter.trigger_lwt()  # must not raise


def test_mqtt_subscriptions_property():
    adapter = MqttAdapter(broker="b:1")
    adapter.subscribe("a/b", lambda t, p: None)
    adapter.subscribe("c/d", lambda t, p: None)
    assert set(adapter.subscriptions) == {"a/b", "c/d"}


def test_mqtt_query_remote_state():
    adapter = MqttAdapter(broker="b:1")
    state = adapter.query_remote_state({})
    assert state["broker"] == "b:1"
    assert "subscriptions" in state


# ===========================================================================
# ExecutorOptions
# ===========================================================================

def test_executor_options_empty():
    opts = ExecutorOptions()
    assert opts.encode() == b""
    assert len(opts) == 0


def test_executor_options_lz_receive():
    opts = ExecutorOptions()
    opts.add_lz_receive(200_000)
    encoded = opts.encode()
    assert len(encoded) > 0
    assert len(opts) == 1


def test_executor_options_native_drop():
    opts = ExecutorOptions()
    opts.add_native_drop(1_000_000_000_000_000, "aa" * 20)
    encoded = opts.encode()
    assert len(encoded) > 0


def test_executor_options_chain():
    opts = ExecutorOptions()
    opts.add_lz_receive(100_000).add_native_drop(1, "bb" * 20)
    assert len(opts) == 2
    encoded = opts.encode()
    assert len(encoded) > 0


def test_executor_options_type_prefix():
    opts = ExecutorOptions()
    opts.add_lz_receive(50_000)
    raw = opts.encode()
    # First 2 bytes = uint16 type = 1
    t = struct.unpack(">H", raw[:2])[0]
    assert t == LzOptionType.LZRECEIVE


# ===========================================================================
# MessagingFee
# ===========================================================================

def test_messaging_fee_total():
    fee = MessagingFee(native_fee=1_000, lz_token_fee=500)
    assert fee.total_wei == 1_500


def test_messaging_fee_defaults():
    fee = MessagingFee()
    assert fee.total_wei == 0


# ===========================================================================
# SendParam
# ===========================================================================

def test_send_param_to_dict():
    sp = SendParam(dst_eid=30110, to="0xabc", amount_ld=1_000)
    d = sp.to_dict()
    assert d["dst_eid"] == 30110
    assert d["to"] == "0xabc"


def test_send_param_extra_options_hex():
    opts = ExecutorOptions()
    opts.add_lz_receive(100_000)
    sp = SendParam(extra_options=opts.encode())
    d = sp.to_dict()
    assert isinstance(d["extra_options"], str)


# ===========================================================================
# Origin
# ===========================================================================

def test_origin_fields():
    o = Origin(src_eid=30101, sender="0xAAA", nonce=5)
    assert o.src_eid == 30101
    assert o.nonce == 5


# ===========================================================================
# LzPacket — extended
# ===========================================================================

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
    assert pkt1.keccak_guid != pkt2.keccak_guid


def test_lz_packet_to_origin():
    pkt = LzPacket(src_eid=30101, sender="0xSEND", nonce=7)
    origin = pkt.to_origin()
    assert isinstance(origin, Origin)
    assert origin.src_eid == 30101
    assert origin.nonce == 7


def test_lz_packet_has_compose_msg():
    pkt = LzPacket(compose_msg=b"compose-payload")
    assert pkt.compose_msg == b"compose-payload"


# ===========================================================================
# LayerZeroAdapter — basic dispatch
# ===========================================================================

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
    assert len(hooked[0].message) > 0


def test_lz_send_hook_raises_gives_error():
    def bad_hook(pkt):
        raise OSError("endpoint down")

    adapter = LayerZeroAdapter(send_hook=bad_hook)
    result = adapter.dispatch(_env())
    assert not result.ok
    assert "endpoint down" in result.error


def test_lz_failed_packets_recorded():
    adapter = LayerZeroAdapter(send_hook=lambda pkt: (_ for _ in ()).throw(OSError()))
    adapter.dispatch(_env())
    assert len(adapter.failed_packets) == 1


def test_lz_result_metadata_contains_guid():
    adapter = LayerZeroAdapter()
    result = adapter.dispatch(_env())
    assert "lz_guid" in result.metadata
    assert "nonce" in result.metadata


def test_lz_result_metadata_has_compose_flag():
    adapter = LayerZeroAdapter()
    result = adapter.dispatch(_env(), compose_msg=b"compose")
    assert result.metadata["has_compose"] is True


def test_lz_dispatch_with_options():
    opts = ExecutorOptions()
    opts.add_lz_receive(200_000)
    adapter = LayerZeroAdapter()
    result = adapter.dispatch(_env(), options=opts.encode())
    assert result.ok
    assert adapter.sent_packets[0].options == opts.encode()


# ===========================================================================
# LayerZeroAdapter — DVN simulation
# ===========================================================================

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


def test_lz_dvn_proof_unique_per_dvn():
    adapter = LayerZeroAdapter(dvn_count=2)
    pkt = LzPacket(nonce=1, message=b"x")
    pkt.guid = pkt.keccak_guid
    receipts = adapter.simulate_dvn_receive(pkt)
    assert receipts[0].proof != receipts[1].proof


# ===========================================================================
# LayerZeroAdapter — lz_receive
# ===========================================================================

def test_lz_receive_basic():
    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    adapter.dispatch(_env())
    pkt = adapter.sent_packets[0]
    summary = adapter.lz_receive(pkt)
    assert summary["payload_len"] > 0
    assert summary["composed"] is False


def test_lz_receive_with_hook():
    origins_seen: list[Origin] = []

    def recv_hook(origin: Origin, msg: bytes) -> None:
        origins_seen.append(origin)

    adapter = LayerZeroAdapter(receive_hook=recv_hook)
    adapter.dispatch(_env())
    pkt = adapter.sent_packets[0]
    adapter.lz_receive(pkt)
    assert len(origins_seen) == 1
    assert origins_seen[0].src_eid == adapter.src_eid


def test_lz_receive_with_compose():
    adapter = LayerZeroAdapter()
    adapter.dispatch(_env(), compose_msg=b"compose-payload")
    pkt = adapter.sent_packets[0]
    summary = adapter.lz_receive(pkt)
    assert summary["composed"] is True
    assert len(adapter.compose_log) == 1


def test_lz_received_packets_property():
    adapter = LayerZeroAdapter()
    adapter.dispatch(_env())
    pkt = adapter.sent_packets[0]
    adapter.lz_receive(pkt)
    assert len(adapter.received_packets) == 1


# ===========================================================================
# LayerZeroAdapter — quote_send
# ===========================================================================

def test_quote_send_native_only():
    adapter = LayerZeroAdapter(native_fee_wei=1_000_000_000_000_000)
    sp = SendParam(dst_eid=30110, to="0xabc", amount_ld=100)
    fee = adapter.quote_send(sp)
    assert isinstance(fee, MessagingFee)
    assert fee.native_fee >= 1_000_000_000_000_000
    assert fee.lz_token_fee == 0


def test_quote_send_lz_token_split():
    adapter = LayerZeroAdapter(native_fee_wei=1_000_000_000_000_000)
    sp = SendParam(dst_eid=30110)
    fee = adapter.quote_send(sp, pay_in_lz_token=True)
    assert fee.lz_token_fee > 0
    assert fee.native_fee + fee.lz_token_fee == fee.total_wei


def test_quote_send_compose_increases_fee():
    adapter = LayerZeroAdapter(native_fee_wei=1_000_000_000_000_000)
    sp_simple = SendParam(dst_eid=30110)
    sp_compose = SendParam(dst_eid=30110, compose_msg=b"x" * 100)
    fee_simple  = adapter.quote_send(sp_simple)
    fee_compose = adapter.quote_send(sp_compose)
    assert fee_compose.native_fee > fee_simple.native_fee


# ===========================================================================
# LayerZeroAdapter — lzRead
# ===========================================================================

def test_lz_read_returns_response():
    adapter = LayerZeroAdapter()
    resp = adapter.lz_read(30110, "0xTarget", b"\x01\x02")
    assert isinstance(resp, LzReadResponse)
    assert resp.success
    assert len(resp.result) > 0


def test_lz_read_deterministic():
    adapter = LayerZeroAdapter()
    r1 = adapter.lz_read(30110, "0xA", b"data")
    r2 = adapter.lz_read(30110, "0xA", b"data")
    assert r1.result == r2.result


def test_lz_read_different_call_data():
    adapter = LayerZeroAdapter()
    r1 = adapter.lz_read(30110, "0xA", b"data1")
    r2 = adapter.lz_read(30110, "0xA", b"data2")
    assert r1.result != r2.result


def test_lz_read_logged():
    adapter = LayerZeroAdapter()
    adapter.lz_read(30110, "0xA")
    assert len(adapter.read_log) == 1


# ===========================================================================
# LayerZeroAdapter — query_remote_state (delegates to lzRead)
# ===========================================================================

def test_lz_query_remote_state():
    adapter = LayerZeroAdapter(src_eid=30101, dst_eid=30110)
    adapter.dispatch(_env())
    state = adapter.query_remote_state({"block": 999})
    assert state["packets_sent"] == 1
    assert state["src_eid"] == 30101
    assert "lzread" in state


def test_lz_query_remote_state_custom_target():
    adapter = LayerZeroAdapter()
    state = adapter.query_remote_state({
        "target_eid": 30184, "target_address": "0xBase"
    })
    assert state["lzread"]["success"] is True


# ===========================================================================
# LayerZeroAdapter — retry_message
# ===========================================================================

def test_retry_failed_message_ok():
    adapter = LayerZeroAdapter(send_hook=lambda pkt: (_ for _ in ()).throw(OSError("fail")))
    adapter.dispatch(_env())
    assert len(adapter.failed_packets) == 1
    failed_pkt = adapter.failed_packets[0]
    # Clear hook so retry succeeds
    adapter._send_hook = None
    result = adapter.retry_message(failed_pkt)
    assert result.ok
    assert len(adapter.failed_packets) == 0


def test_retry_still_fails():
    def always_fail(pkt):
        raise OSError("still down")

    adapter = LayerZeroAdapter(send_hook=always_fail)
    adapter.dispatch(_env())
    pkt = adapter.failed_packets[0]
    result = adapter.retry_message(pkt)
    assert not result.ok


def test_retry_metadata_has_retry_flag():
    adapter = LayerZeroAdapter()
    pkt = LzPacket(nonce=1, message=b"test")
    pkt.guid = pkt.keccak_guid
    result = adapter.retry_message(pkt)
    assert result.metadata.get("retry") is True


# ===========================================================================
# ArbitrageChannel
# ===========================================================================

def test_arb_channel_publish_dispatch():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, asset="ETH/USDC")
    result = ch.publish_price(2000.0)
    assert result.ok
    assert len(lz.sent_packets) == 1


def test_arb_channel_no_opportunity_single_price():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=10.0)
    ch.publish_price(2000.0, is_source=True)
    assert len(ch.opportunities) == 0


def test_arb_channel_opportunity_detected():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=10.0)
    ch.publish_price(2000.0, is_source=True)
    ch.publish_price(2050.0, is_source=False)
    assert len(ch.opportunities) == 1
    opp = ch.opportunities[0]
    assert opp.asset == "ETH/USDC"
    assert opp.spread_bps > 0


def test_arb_channel_below_threshold_no_opportunity():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=100.0)
    ch.publish_price(2000.0, is_source=True)
    ch.publish_price(2001.0, is_source=False)   # 5 bps
    assert len(ch.opportunities) == 0


def test_arb_channel_negative_spread():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=10.0)
    ch.publish_price(2050.0, is_source=True)
    ch.publish_price(2000.0, is_source=False)   # negative spread, |bps| > 10
    assert len(ch.opportunities) == 1
    assert ch.opportunities[0].spread_bps < 0


def test_arb_channel_lz_guid_in_opportunity():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=5.0)
    ch.publish_price(1000.0, is_source=True)
    ch.publish_price(1002.0, is_source=False)   # 20 bps
    opp = ch.opportunities[0]
    assert opp.lz_guid != ""


def test_arb_channel_as_float32_feature_no_prices():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz)
    raw = ch.as_float32_feature()
    assert len(raw) == 4
    value = struct.unpack("f", raw)[0]
    assert value == 0.0


def test_arb_channel_as_float32_feature_with_prices():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=0.0)
    ch.publish_price(1000.0, is_source=True)
    ch.publish_price(1010.0, is_source=False)
    raw = ch.as_float32_feature()
    assert len(raw) == 4
    value = struct.unpack("f", raw)[0]
    assert value > 0.0


def test_arb_channel_repr():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, asset="BTC/USDC")
    assert "BTC/USDC" in repr(ch)


def test_arb_channel_multiple_opportunities():
    lz = LayerZeroAdapter()
    ch = ArbitrageChannel(lz, min_spread_bps=5.0)
    for i in range(5):
        ch.publish_price(1000.0 + i, is_source=True)
        ch.publish_price(1005.0 + i, is_source=False)
    assert len(ch.opportunities) >= 5

