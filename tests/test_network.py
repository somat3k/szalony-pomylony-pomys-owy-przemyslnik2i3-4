"""Tests for the network layer."""

import pytest
from hololang.network.api import Channel, Protocol, Message, ApiEndpoint
from hololang.network.websocket import WebSocketConnection, WebSocketServer
from hololang.network.webhook import Webhook, GrpcChannel, WebhookEvent


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

def test_message_creation():
    m = Message("test", {"key": "value"})
    assert m.topic == "test"
    assert m.payload == {"key": "value"}
    assert m.msg_id


def test_message_to_dict():
    m = Message("t", 42)
    d = m.to_dict()
    assert d["topic"] == "t"
    assert d["payload"] == 42
    assert "timestamp" in d


def test_message_to_json():
    m = Message("t", 1)
    j = m.to_json()
    assert '"topic"' in j


def test_message_from_dict():
    m = Message.from_dict({"topic": "x", "payload": 99})
    assert m.topic == "x"
    assert m.payload == 99


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

def test_channel_default_port():
    ch = Channel("grpc_chan", protocol="grpc")
    assert ch.port == 50051


def test_channel_ws_port():
    ch = Channel("ws_chan", protocol="ws")
    assert ch.port == 8080


def test_channel_open_close():
    ch = Channel("ch", protocol="http")
    assert not ch.is_open()
    ch.open()
    assert ch.is_open()
    ch.close()
    assert not ch.is_open()


def test_channel_emit_listen():
    ch = Channel("evt")
    ch.open()
    received = []
    ch.listen(lambda msg: received.append(msg))
    msg = ch.emit("hello", topic="greet")
    assert msg.payload == "hello"
    assert len(received) == 1


def test_channel_receive_all():
    ch = Channel("recv")
    ch.open()
    ch.emit(1)
    ch.emit(2)
    msgs = ch.receive_all()
    assert len(msgs) == 2
    # After receive_all inbox is cleared
    assert ch.receive_all() == []


def test_channel_url():
    ch = Channel("u", protocol="grpc", host="myhost", port=9999)
    assert "myhost" in ch.url
    assert "9999" in ch.url


# ---------------------------------------------------------------------------
# ApiEndpoint
# ---------------------------------------------------------------------------

def test_api_route():
    api = ApiEndpoint("test_api")
    route = api.get("/status")
    assert route.path == "/api/v1/status"


def test_api_call_no_handler():
    api = ApiEndpoint("no_handler")
    result = api.call("GET", "/info")
    assert "error" in result


def test_api_call_with_handler():
    api = ApiEndpoint("with_handler")
    api.get("/hello", handler=lambda body, params: {"msg": "hi"})
    result = api.call("GET", "/hello")
    assert result == {"msg": "hi"}


def test_api_post():
    api = ApiEndpoint("post_test")
    api.post("/data", handler=lambda body, params: {"received": body})
    result = api.call("POST", "/data", body={"x": 1})
    assert result["received"] == {"x": 1}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

def test_websocket_connect():
    ws = WebSocketConnection()
    ws.connect("ws://localhost:8080")
    assert ws.is_open()


def test_websocket_send():
    ws = WebSocketConnection()
    ws.connect("ws://localhost:8080")
    ws.send("ping")
    sent = ws.sent_all()
    assert len(sent) == 1
    assert sent[0].payload == "ping"


def test_websocket_send_when_closed():
    ws = WebSocketConnection()
    with pytest.raises(ConnectionError):
        ws.send("data")


def test_websocket_receive():
    ws = WebSocketConnection()
    ws.connect()
    ws.receive("server message")
    msgs = ws.receive_all()
    assert msgs[0].payload == "server message"


def test_websocket_on_open():
    events = []
    ws = WebSocketConnection()
    ws.on("open", lambda url="": events.append(("open", url)))
    ws.connect("ws://test")
    assert any(e[0] == "open" for e in events)


def test_websocket_server_broadcast():
    server = WebSocketServer()
    server.start()
    c1 = server.accept()
    c2 = server.accept()
    server.broadcast("hello all")
    assert c1.receive_all()[0].payload == "hello all"
    assert c2.receive_all()[0].payload == "hello all"


def test_websocket_server_connection_count():
    server = WebSocketServer()
    assert server.connection_count() == 0
    server.accept()
    server.accept()
    assert server.connection_count() == 2


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def test_webhook_dispatch():
    wh = Webhook("test_hook", url="http://example.com/hook")
    ev = wh.dispatch("deploy", {"version": "1.0"})
    assert ev.event == "deploy"
    assert len(wh.sent_events()) == 1


def test_webhook_receive():
    wh = Webhook("recv_hook")
    received = []
    wh.on("data_event", received.append)
    ev = wh.receive("data_event", {"key": "val"})
    assert len(received) == 1
    assert received[0].payload == {"key": "val"}


def test_webhook_wildcard_listener():
    wh = Webhook("wildcard")
    all_events = []
    wh.on("*", all_events.append)
    wh.receive("event_a", 1)
    wh.receive("event_b", 2)
    assert len(all_events) == 2


def test_webhook_to_json():
    ev = WebhookEvent(event="test", payload=123)
    j = ev.to_json()
    assert '"test"' in j


# ---------------------------------------------------------------------------
# gRPC channel stub
# ---------------------------------------------------------------------------

def test_grpc_open_close():
    ch = GrpcChannel("localhost")
    ch.open()
    assert ch._open
    ch.close()
    assert not ch._open


def test_grpc_call():
    ch = GrpcChannel("localhost", port=50051)
    ch.open()
    resp = ch.call("GetBeamState", {"device_id": "laser-001"})
    assert resp["status"] == "ok"
    assert resp["method"] == "GetBeamState"


def test_grpc_call_when_closed():
    ch = GrpcChannel("localhost")
    with pytest.raises(ConnectionError):
        ch.call("Any", {})


def test_grpc_address():
    ch = GrpcChannel("myhost", port=1234)
    assert ch.address == "myhost:1234"


def test_grpc_call_history():
    ch = GrpcChannel("localhost")
    ch.open()
    ch.call("MethodA", {})
    ch.call("MethodB", {})
    history = ch.call_history()
    assert len(history) == 2
    assert history[0]["method"] == "MethodA"
