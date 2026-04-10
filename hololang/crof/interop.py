"""Plane F — Interdomain Plane: pluggable domain adapters.

The CROF fabric communicates across domain boundaries through a unified
:class:`DomainAdapter` interface.  Concrete adapters are provided for:

* :class:`GrpcAdapter`       — direct gRPC service-to-service traffic
* :class:`MqttAdapter`       — device / embedded-network messaging
* :class:`LayerZeroAdapter`  — omnichain trust-minimised bridge (LayerZero V2)
* :class:`ArbitrageChannel`  — cross-chain price-feed / opportunity relay

LayerZero V2 OApp integration
------------------------------
The :class:`LayerZeroAdapter` implements the full LayerZero V2 **OApp** pattern:

* **LzPacket** — mirrors the on-chain ``Packet`` struct (src_eid, dst_eid,
  sender, receiver, nonce, guid, message, options).
* **SendParam** — full V2 ``SendParam`` struct (dstEid, to, amountLD,
  minAmountLD, extraOptions, composeMsg, oftCmd).
* **MessagingFee** — V2 fee estimate (nativeFee + lzTokenFee).
* **Origin** — receiving-side origin context (srcEid, sender, nonce).
* **ExecutorOptions** — type-encoded execution options:
  - Type 1 (``LZRECEIVE``) — gas limit for ``_lzReceive``.
  - Type 2 (``NATIVE_DROP``) — amount + receiver for native-token airdrop.
* **LzReadRequest / LzReadResponse** — lzRead cross-chain state queries.
* **DvnVerification** — simulated DVN ``verifyAndAttest`` receipt.

Adapter hooks let tests and production wiring inject transport without
changing adapter logic.  Every adapter exposes:

* ``dispatch(envelope)`` → :class:`DispatchResult`
* ``query_remote_state(query)`` → ``dict``

Each adapter returns a :class:`DispatchResult` that records delivery
status, latency, and any error message.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from hololang.crof.envelope import Envelope


# ---------------------------------------------------------------------------
# DispatchResult
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Outcome of a single cross-domain dispatch.

    Attributes
    ----------
    packet_id:
        Unique identifier for this dispatch attempt.
    adapter_type:
        Name of the adapter that was used (``"grpc"``, ``"mqtt"``, ``"layerzero"``…).
    status:
        ``"dispatched"`` | ``"delivered"`` | ``"failed"`` | ``"queued"``
    latency_ms:
        Measured wall-clock latency in milliseconds.
    error:
        Non-empty string when dispatch failed.
    metadata:
        Adapter-specific extension fields.
    """
    packet_id:    str            = field(default_factory=lambda: str(uuid.uuid4()))
    adapter_type: str            = ""
    status:       str            = "dispatched"
    latency_ms:   float          = 0.0
    error:        str            = ""
    metadata:     dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and self.status != "failed"

    def __repr__(self) -> str:
        return (
            f"DispatchResult(adapter={self.adapter_type!r}, "
            f"status={self.status!r}, ok={self.ok})"
        )


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class DomainAdapter(ABC):
    """Abstract base class for all CROF domain adapters."""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        ...

    @abstractmethod
    def dispatch(self, envelope: Envelope) -> DispatchResult:
        """Send *envelope* to the remote domain."""

    @abstractmethod
    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        """Query state from the remote domain."""


# ---------------------------------------------------------------------------
# gRPC adapter
# ---------------------------------------------------------------------------

class GrpcAdapter(DomainAdapter):
    """Direct gRPC service-to-service adapter.

    In production this would use the generated gRPC stubs from the CROF
    ``.proto`` definitions.  Here it simulates the transport layer.

    Parameters
    ----------
    endpoint:
        gRPC target in ``"host:port"`` format.
    service:
        gRPC service name (e.g. ``"InteropService"``).
    timeout_ms:
        Request timeout in milliseconds.
    max_retries:
        Maximum number of retry attempts on transient failure (default 3).
    metadata:
        Channel metadata key-value pairs sent as gRPC headers (e.g. auth tokens).
    """

    def __init__(
        self,
        endpoint: str,
        service: str = "InteropService",
        timeout_ms: int = 5000,
        max_retries: int = 3,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.endpoint    = endpoint
        self.service     = service
        self.timeout_ms  = timeout_ms
        self.max_retries = max_retries
        self.metadata    = metadata or {}
        self._call_hook: Callable[[str, bytes], bytes] | None = None
        self._interceptors: list[Callable[[str, bytes], bytes]] = []
        self._call_log: list[dict[str, Any]] = []

    def register_call_hook(self, hook: Callable[[str, bytes], bytes]) -> None:
        """Register a callback that stands in for the real gRPC channel."""
        self._call_hook = hook

    def add_interceptor(self, fn: Callable[[str, bytes], bytes]) -> None:
        """Append a unary interceptor invoked before the call hook."""
        self._interceptors.append(fn)

    @property
    def adapter_type(self) -> str:
        return "grpc"

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Ordered record of every RPC call attempted."""
        return list(self._call_log)

    def _rpc(self, method: str, payload: bytes) -> tuple[bytes, str]:
        """Execute one RPC through interceptors and the call hook."""
        current = payload
        for fn in self._interceptors:
            try:
                current = fn(method, current)
            except Exception as exc:  # noqa: BLE001
                return b"", str(exc)
        if self._call_hook is not None:
            try:
                result = self._call_hook(method, current)
                return result if result else b"", ""
            except Exception as exc:  # noqa: BLE001
                return b"", str(exc)
        return b"", ""

    def dispatch(self, envelope: Envelope) -> DispatchResult:
        t0      = time.perf_counter()
        payload = envelope.to_json().encode()
        method  = f"{self.service}.DispatchInterdomain"
        error   = ""
        for attempt in range(max(1, self.max_retries)):
            _, error = self._rpc(method, payload)
            self._call_log.append({
                "method": method, "attempt": attempt + 1, "error": error,
            })
            if not error:
                break
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {
                "endpoint": self.endpoint,
                "service":  self.service,
                "attempts": len(self._call_log),
            },
        )

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        method = f"{self.service}.QueryRemoteState"
        raw, _ = self._rpc(method, json.dumps(query).encode())
        self._call_log.append({"method": method, "attempt": 1, "error": ""})
        if raw:
            try:
                return json.loads(raw)
            except Exception:  # noqa: BLE001
                pass
        return {"status": "simulated", "endpoint": self.endpoint}

    def health_check(self) -> bool:
        """Perform a gRPC health check (``grpc.health.v1.Health.Check``).

        Returns ``True`` when the call hook is registered and returns
        without raising, or when no hook is set (simulated healthy).
        """
        if self._call_hook is None:
            return True
        _, err = self._rpc("grpc.health.v1.Health.Check", b"{}")
        return not err


# ---------------------------------------------------------------------------
# MQTT adapter
# ---------------------------------------------------------------------------

@dataclass
class MqttRetainedMessage:
    """A retained message stored for a topic."""
    topic:     str
    payload:   bytes
    qos:       int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MqttWillMessage:
    """Last-will-and-testament message sent on unexpected disconnect."""
    topic:   str
    payload: bytes
    qos:     int = 1
    retain:  bool = False


class MqttAdapter(DomainAdapter):
    """Device / embedded-network adapter using the MQTT pub/sub model.

    Parameters
    ----------
    broker:
        MQTT broker address in ``"host:port"`` format.
    topic_prefix:
        Prefix for all published topics (e.g. ``"crof/domain/edge"``).
    qos:
        Default MQTT QoS level (0, 1, or 2).
    will:
        Optional :class:`MqttWillMessage` (Last Will and Testament).
    """

    def __init__(
        self,
        broker: str,
        topic_prefix: str = "crof",
        qos: int = 1,
        will: MqttWillMessage | None = None,
    ) -> None:
        self.broker       = broker
        self.topic_prefix = topic_prefix
        self.qos          = qos
        self.will         = will
        self._publish_hook: Callable[[str, bytes, int], None] | None = None
        self._subscriptions: dict[str, list[Callable[[str, bytes], None]]] = {}
        self._retained: dict[str, MqttRetainedMessage] = {}
        self._delivery_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    def register_publish_hook(
        self, hook: Callable[[str, bytes, int], None]
    ) -> None:
        """Register a callback that stands in for the real MQTT client."""
        self._publish_hook = hook

    def subscribe(
        self,
        topic_filter: str,
        callback: Callable[[str, bytes], None],
    ) -> None:
        """Subscribe a *callback* to messages matching *topic_filter*."""
        self._subscriptions.setdefault(topic_filter, []).append(callback)

    def unsubscribe(self, topic_filter: str) -> None:
        """Remove all callbacks for *topic_filter*."""
        self._subscriptions.pop(topic_filter, None)

    @property
    def subscriptions(self) -> list[str]:
        """Active subscription topic filters."""
        return list(self._subscriptions)

    @property
    def retained(self) -> dict[str, MqttRetainedMessage]:
        """Currently retained messages keyed by topic."""
        return dict(self._retained)

    @property
    def delivery_counts(self) -> dict[int, int]:
        """Messages delivered per QoS level."""
        return dict(self._delivery_counts)

    def _match_topic(self, topic_filter: str, topic: str) -> bool:
        """MQTT wildcard matching (``#`` and ``+`` supported)."""
        if topic_filter == topic:
            return True
        f_parts = topic_filter.split("/")
        t_parts = topic.split("/")
        fi = ti = 0
        while fi < len(f_parts) and ti < len(t_parts):
            if f_parts[fi] == "#":
                return True
            if f_parts[fi] != "+" and f_parts[fi] != t_parts[ti]:
                return False
            fi += 1
            ti += 1
        return fi == len(f_parts) and ti == len(t_parts)

    def _deliver_to_subscribers(self, topic: str, payload: bytes) -> None:
        for filt, callbacks in self._subscriptions.items():
            if self._match_topic(filt, topic):
                for cb in callbacks:
                    try:
                        cb(topic, payload)
                    except Exception:  # noqa: BLE001
                        pass

    @property
    def adapter_type(self) -> str:
        return "mqtt"

    def dispatch(
        self,
        envelope: Envelope,
        retain: bool = False,
    ) -> DispatchResult:
        topic   = (
            f"{self.topic_prefix}/{envelope.domain_id}/{envelope.operation_type}"
        )
        payload = envelope.to_json().encode()
        t0      = time.perf_counter()
        error   = ""
        if self._publish_hook is not None:
            try:
                self._publish_hook(topic, payload, self.qos)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        if not error:
            self._delivery_counts[self.qos] = (
                self._delivery_counts.get(self.qos, 0) + 1
            )
            if retain:
                self._retained[topic] = MqttRetainedMessage(
                    topic=topic, payload=payload, qos=self.qos
                )
            self._deliver_to_subscribers(topic, payload)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {"broker": self.broker, "topic": topic, "qos": self.qos},
        )

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        return {
            "status":       "simulated",
            "broker":       self.broker,
            "subscriptions": len(self._subscriptions),
            "retained":     len(self._retained),
        }

    def trigger_lwt(self) -> None:
        """Simulate an unexpected disconnect — publish the LWT message."""
        if self.will is None:
            return
        payload = self.will.payload
        if self.will.retain:
            self._retained[self.will.topic] = MqttRetainedMessage(
                topic=self.will.topic, payload=payload, qos=self.will.qos
            )
        self._deliver_to_subscribers(self.will.topic, payload)


# ---------------------------------------------------------------------------
# LayerZero V2 data structures
# ---------------------------------------------------------------------------

class LzOptionType(IntEnum):
    """Executor option type codes (LayerZero V2 spec §2.3)."""
    LZRECEIVE  = 1   # gas limit for _lzReceive on destination
    NATIVE_DROP = 2  # airdrop native token to a receiver on destination


@dataclass
class ExecutorOptions:
    """Builder for LayerZero V2 executor options bytes.

    Each option is encoded as:
    ``[uint16 type][uint128 gas]`` (type 1)  or
    ``[uint16 type][uint128 amount][bytes32 receiver]`` (type 2).

    Use :meth:`add_lz_receive` and :meth:`add_native_drop` to compose
    options, then call :meth:`encode` to obtain the final ``bytes``.
    """

    _options: list[tuple[int, bytes]] = field(default_factory=list)

    def add_lz_receive(self, gas: int) -> "ExecutorOptions":
        """Add a type-1 (LZRECEIVE) option with the given *gas* limit."""
        # uint16 type (1) + uint128 gas
        raw = struct.pack(">HQ", LzOptionType.LZRECEIVE, gas)
        self._options.append((LzOptionType.LZRECEIVE, raw))
        return self

    def add_native_drop(self, amount: int, receiver_hex: str) -> "ExecutorOptions":
        """Add a type-2 (NATIVE_DROP) option.

        Parameters
        ----------
        amount:
            Native token amount in wei.
        receiver_hex:
            Receiver address as a 40-char hex string (without ``0x``).
        """
        receiver_bytes = bytes.fromhex(receiver_hex.lstrip("0x").zfill(40))
        raw = struct.pack(">HQ", LzOptionType.NATIVE_DROP, amount) + receiver_bytes
        self._options.append((LzOptionType.NATIVE_DROP, raw))
        return self

    def encode(self) -> bytes:
        """Return the concatenated binary option blob."""
        return b"".join(raw for _, raw in self._options)

    def __len__(self) -> int:
        return len(self._options)


@dataclass
class MessagingFee:
    """LayerZero V2 ``MessagingFee`` struct returned by ``quoteSend``.

    Attributes
    ----------
    native_fee:
        Fee paid in the source chain's native token (wei).
    lz_token_fee:
        Optional fee paid in the LayerZero token (wei).
    """
    native_fee:    int = 0
    lz_token_fee:  int = 0

    @property
    def total_wei(self) -> int:
        return self.native_fee + self.lz_token_fee


@dataclass
class SendParam:
    """LayerZero V2 ``SendParam`` struct (OFT-compatible).

    Fields
    ------
    dst_eid:
        Destination chain endpoint ID.
    to:
        Recipient address on the destination chain (bytes32 hex).
    amount_ld:
        Amount in local decimals to send.
    min_amount_ld:
        Minimum acceptable amount after slippage.
    extra_options:
        Encoded executor options (:meth:`ExecutorOptions.encode`).
    compose_msg:
        Compose message bytes (empty bytes = no compose).
    oft_cmd:
        OFT-specific command bytes.
    """
    dst_eid:       int   = 0
    to:            str   = ""
    amount_ld:     int   = 0
    min_amount_ld: int   = 0
    extra_options: bytes = b""
    compose_msg:   bytes = b""
    oft_cmd:       bytes = b""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dst_eid":       self.dst_eid,
            "to":            self.to,
            "amount_ld":     self.amount_ld,
            "min_amount_ld": self.min_amount_ld,
            "extra_options": self.extra_options.hex(),
            "compose_msg":   self.compose_msg.hex(),
            "oft_cmd":       self.oft_cmd.hex(),
        }


@dataclass
class Origin:
    """LayerZero V2 ``Origin`` struct supplied to ``_lzReceive``.

    Attributes
    ----------
    src_eid:
        Source chain endpoint ID.
    sender:
        Sending OApp address (bytes32 hex).
    nonce:
        Packet nonce (monotonically increasing per sender→receiver path).
    """
    src_eid: int = 0
    sender:  str = ""
    nonce:   int = 0


@dataclass
class LzReadRequest:
    """A cross-chain ``lzRead`` state query (LayerZero V2 lzRead module).

    Attributes
    ----------
    request_id:
        Unique request identifier.
    src_eid:
        Source chain endpoint ID where the read is initiated.
    target_eid:
        Target chain endpoint ID whose state is being read.
    target_address:
        Contract address on the target chain.
    call_data:
        ABI-encoded call data (empty = balance read).
    """
    request_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    src_eid:        int   = 0
    target_eid:     int   = 0
    target_address: str   = ""
    call_data:      bytes = b""


@dataclass
class LzReadResponse:
    """Response to an :class:`LzReadRequest`."""
    request_id: str
    success:    bool
    result:     bytes = b""
    error:      str   = ""


@dataclass
class LzPacket:
    """LayerZero V2 message packet (mirrors the on-chain ``Packet`` struct).

    Fields
    ------
    src_eid:
        Source endpoint ID (LayerZero chain ID, e.g. ``30101`` for Ethereum).
    dst_eid:
        Destination endpoint ID.
    sender:
        Bytes32 address of the sending OApp.
    receiver:
        Bytes32 address of the receiving OApp.
    nonce:
        Monotonically increasing per-sender-receiver nonce.
    guid:
        Globally unique packet identifier (keccak256 of packet header).
    message:
        Application-level payload bytes.
    options:
        Encoded execution options (:meth:`ExecutorOptions.encode`).
    compose_msg:
        Optional compose message (non-empty triggers ``lzCompose``).
    """
    src_eid:     int   = 0
    dst_eid:     int   = 0
    sender:      str   = ""
    receiver:    str   = ""
    nonce:       int   = 0
    guid:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    message:     bytes = b""
    options:     bytes = b""
    compose_msg: bytes = b""

    def encode(self) -> bytes:
        """Return a canonical binary encoding of the packet header."""
        header = json.dumps({
            "src_eid":  self.src_eid,
            "dst_eid":  self.dst_eid,
            "sender":   self.sender,
            "receiver": self.receiver,
            "nonce":    self.nonce,
        }, sort_keys=True, separators=(",", ":")).encode()
        return header + self.message

    @property
    def keccak_guid(self) -> str:
        """Return the keccak-256 GUID of the packet header (SHA-256 here)."""
        return hashlib.sha256(self.encode()).hexdigest()

    def to_origin(self) -> Origin:
        """Return an :class:`Origin` representing the source of this packet."""
        return Origin(src_eid=self.src_eid, sender=self.sender, nonce=self.nonce)


@dataclass
class DvnVerification:
    """Simulated DVN (Decentralised Verification Network) receipt."""
    packet_guid: str
    dvn_id:      str
    verified:    bool
    timestamp:   int   = field(default_factory=lambda: int(time.time()))
    proof:       bytes = b""


# ---------------------------------------------------------------------------
# LayerZero adapter (Class-6 Interop Node — blockchain bridge)
# ---------------------------------------------------------------------------

class LayerZeroAdapter(DomainAdapter):
    """Trust-minimised omnichain messaging adapter using LayerZero V2 OApp.

    The adapter wraps a CROF :class:`Envelope` in an :class:`LzPacket`
    following the LayerZero V2 ``_lzSend`` / ``_lzReceive`` pattern.

    Parameters
    ----------
    src_eid:
        Source chain endpoint ID (e.g. ``30101`` = Ethereum mainnet).
    dst_eid:
        Destination chain endpoint ID (e.g. ``30110`` = Arbitrum One).
    sender:
        OApp contract address (hex string) on the source chain.
    receiver:
        OApp contract address (hex string) on the destination chain.
    endpoint_url:
        URL of the LayerZero Endpoint (for production web3.py wiring).
    dvn_count:
        Number of DVN confirmations to simulate (≥ 1).
    send_hook:
        Optional callable ``(LzPacket) -> None`` for production endpoint wiring.
    receive_hook:
        Optional callable ``(Origin, bytes) -> None`` called by
        :meth:`lz_receive` to simulate the ``_lzReceive`` OApp override.
    native_fee_wei:
        Simulated native-token fee returned by :meth:`quote_send` (default 1e15 wei).
    """

    def __init__(
        self,
        src_eid:         int = 30101,
        dst_eid:         int = 30110,
        sender:          str = "0x" + "00" * 20,
        receiver:        str = "0x" + "00" * 20,
        endpoint_url:    str = "",
        dvn_count:       int = 1,
        send_hook:       Callable[["LzPacket"], None] | None = None,
        receive_hook:    Callable[["Origin", bytes], None] | None = None,
        native_fee_wei:  int = 1_000_000_000_000_000,   # 0.001 ETH
    ) -> None:
        self.src_eid        = src_eid
        self.dst_eid        = dst_eid
        self.sender         = sender
        self.receiver       = receiver
        self.endpoint_url   = endpoint_url
        self.dvn_count      = dvn_count
        self._send_hook     = send_hook
        self._receive_hook  = receive_hook
        self.native_fee_wei = native_fee_wei
        self._nonce:        int = 0
        self._sent:         list[LzPacket] = []
        self._received:     list[tuple[Origin, bytes]] = []
        self._dvn_receipts: list[DvnVerification] = []
        self._failed:       list[LzPacket] = []
        self._compose_log:  list[dict[str, Any]] = []
        self._read_log:     list[LzReadResponse] = []

    @property
    def adapter_type(self) -> str:
        return "layerzero"

    # ------------------------------------------------------------------
    # Fee estimation — quoteSend
    # ------------------------------------------------------------------

    def quote_send(
        self,
        send_param: SendParam,
        pay_in_lz_token: bool = False,
    ) -> MessagingFee:
        """Simulate the OApp ``quoteSend`` fee estimation call.

        Returns a :class:`MessagingFee` with a deterministic simulated fee
        proportional to the payload size and destination EID.

        Parameters
        ----------
        send_param:
            The send parameters used for the quote.
        pay_in_lz_token:
            When ``True``, the fee is split between native and LZ token.
        """
        # Simulated fee: base + 1 gwei per byte of compose_msg
        base = self.native_fee_wei
        extra = len(send_param.compose_msg) * 1_000_000_000
        total = base + extra
        if pay_in_lz_token:
            lz_part = total // 4
            return MessagingFee(native_fee=total - lz_part, lz_token_fee=lz_part)
        return MessagingFee(native_fee=total, lz_token_fee=0)

    # ------------------------------------------------------------------
    # OApp _lzSend equivalent
    # ------------------------------------------------------------------

    def _build_packet(
        self,
        envelope: Envelope,
        options: bytes = b"",
        compose_msg: bytes = b"",
    ) -> LzPacket:
        self._nonce += 1
        msg = envelope.to_json().encode()
        pkt = LzPacket(
            src_eid     = self.src_eid,
            dst_eid     = self.dst_eid,
            sender      = self.sender,
            receiver    = self.receiver,
            nonce       = self._nonce,
            message     = msg,
            options     = options,
            compose_msg = compose_msg,
        )
        pkt.guid = pkt.keccak_guid
        return pkt

    def dispatch(
        self,
        envelope: Envelope,
        options: bytes = b"",
        compose_msg: bytes = b"",
    ) -> DispatchResult:
        """Package *envelope* as an :class:`LzPacket` and dispatch it.

        In production this calls the LayerZero Endpoint's ``send()`` function.
        Here it invokes the optional ``send_hook`` and records the packet.

        Parameters
        ----------
        envelope:
            The CROF envelope to send.
        options:
            Pre-encoded :class:`ExecutorOptions` bytes.
        compose_msg:
            Optional compose payload triggering ``lzCompose`` on the receiver.
        """
        t0  = time.perf_counter()
        pkt = self._build_packet(envelope, options=options, compose_msg=compose_msg)
        error = ""
        try:
            if self._send_hook is not None:
                self._send_hook(pkt)
            self._sent.append(pkt)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self._failed.append(pkt)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            packet_id    = pkt.guid,
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {
                "src_eid":     self.src_eid,
                "dst_eid":     self.dst_eid,
                "nonce":       pkt.nonce,
                "lz_guid":     pkt.guid,
                "has_compose": bool(compose_msg),
            },
        )

    # ------------------------------------------------------------------
    # OApp _lzReceive simulation (destination side)
    # ------------------------------------------------------------------

    def lz_receive(self, packet: LzPacket) -> dict[str, Any]:
        """Simulate the OApp ``_lzReceive`` entry point on the destination chain.

        Steps
        -----
        1. Extract the :class:`Origin` from the packet.
        2. Invoke :attr:`_receive_hook` when registered.
        3. Record the received payload.
        4. When the packet carries a ``compose_msg``, simulate ``lzCompose``.

        Returns
        -------
        dict
            Summary with ``origin``, ``payload_len``, and ``composed``.
        """
        origin = packet.to_origin()
        composed = False
        try:
            if self._receive_hook is not None:
                self._receive_hook(origin, packet.message)
            self._received.append((origin, packet.message))
            if packet.compose_msg:
                composed = True
                self._compose_log.append({
                    "guid":        packet.guid,
                    "compose_msg": packet.compose_msg.hex(),
                    "from_eid":    origin.src_eid,
                })
        except Exception:  # noqa: BLE001
            pass
        return {
            "origin":      {"src_eid": origin.src_eid, "nonce": origin.nonce},
            "payload_len": len(packet.message),
            "composed":    composed,
        }

    # ------------------------------------------------------------------
    # DVN simulation (_lzReceive path)
    # ------------------------------------------------------------------

    def simulate_dvn_receive(self, packet: LzPacket) -> list[DvnVerification]:
        """Simulate DVN ``assignJob`` / ``verifyAndAttest`` receipts.

        In production the LayerZero DVN network submits these as
        on-chain transactions to the receiving Endpoint.
        This method generates *dvn_count* synthetic receipts.
        """
        receipts: list[DvnVerification] = []
        for i in range(self.dvn_count):
            proof = hashlib.sha256(
                f"dvn-{i}:{packet.guid}".encode()
            ).digest()
            receipt = DvnVerification(
                packet_guid = packet.guid,
                dvn_id      = f"dvn-{i}",
                verified    = True,
                proof       = proof,
            )
            receipts.append(receipt)
            self._dvn_receipts.append(receipt)
        return receipts

    # ------------------------------------------------------------------
    # lzRead — cross-chain state queries
    # ------------------------------------------------------------------

    def lz_read(
        self,
        target_eid: int,
        target_address: str,
        call_data: bytes = b"",
    ) -> LzReadResponse:
        """Issue a simulated lzRead cross-chain state query.

        In production, the LayerZero lzRead module dispatches an
        ``assignJob`` to read DVNs that call the target contract and
        attest the result back to the source chain.

        Returns a :class:`LzReadResponse` with a deterministic simulated
        result (SHA-256 hash of the call data).
        """
        req = LzReadRequest(
            src_eid        = self.src_eid,
            target_eid     = target_eid,
            target_address = target_address,
            call_data      = call_data,
        )
        result = hashlib.sha256(
            f"lzread:{target_eid}:{target_address}".encode() + call_data
        ).digest()
        resp = LzReadResponse(request_id=req.request_id, success=True, result=result)
        self._read_log.append(resp)
        return resp

    # ------------------------------------------------------------------
    # Retry mechanism
    # ------------------------------------------------------------------

    def retry_message(self, packet: LzPacket) -> DispatchResult:
        """Retry a previously failed packet.

        The packet is re-submitted through the send hook (if registered).
        If it succeeds it is removed from :attr:`failed_packets`.
        """
        t0    = time.perf_counter()
        error = ""
        try:
            if self._send_hook is not None:
                self._send_hook(packet)
            if packet in self._failed:
                self._failed.remove(packet)
            if packet not in self._sent:
                self._sent.append(packet)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            packet_id    = packet.guid,
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {"retry": True, "lz_guid": packet.guid},
        )

    # ------------------------------------------------------------------
    # query_remote_state — lzRead convenience wrapper
    # ------------------------------------------------------------------

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        """Simulate a LayerZero remote state query.

        Accepts an optional ``"target_eid"`` and ``"target_address"`` in
        *query* to delegate to :meth:`lz_read`; otherwise returns a
        summary of the adapter state.
        """
        target_eid     = query.get("target_eid", self.dst_eid)
        target_address = query.get("target_address", self.receiver)
        call_data      = query.get("call_data", b"")
        if isinstance(call_data, str):
            call_data = call_data.encode()
        resp = self.lz_read(target_eid, target_address, call_data)
        return {
            "status":       "simulated",
            "src_eid":      self.src_eid,
            "dst_eid":      self.dst_eid,
            "packets_sent": len(self._sent),
            "dvn_verified": len(self._dvn_receipts),
            "lzread":       {"request_id": resp.request_id, "success": resp.success},
            "query":        query,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sent_packets(self) -> list[LzPacket]:
        return list(self._sent)

    @property
    def received_packets(self) -> list[tuple[Origin, bytes]]:
        return list(self._received)

    @property
    def dvn_receipts(self) -> list[DvnVerification]:
        return list(self._dvn_receipts)

    @property
    def failed_packets(self) -> list[LzPacket]:
        return list(self._failed)

    @property
    def compose_log(self) -> list[dict[str, Any]]:
        return list(self._compose_log)

    @property
    def read_log(self) -> list[LzReadResponse]:
        return list(self._read_log)

    def __repr__(self) -> str:
        return (
            f"LayerZeroAdapter(src_eid={self.src_eid}, dst_eid={self.dst_eid}, "
            f"sent={len(self._sent)}, dvn={len(self._dvn_receipts)})"
        )


# ---------------------------------------------------------------------------
# ArbitrageChannel — cross-chain price-feed / opportunity relay
# ---------------------------------------------------------------------------

@dataclass
class ArbitrageOpportunity:
    """A cross-chain arbitrage price discrepancy detected by the channel.

    Attributes
    ----------
    opportunity_id:
        Unique identifier for this opportunity.
    src_eid:
        Source chain endpoint ID where the lower price is observed.
    dst_eid:
        Destination chain endpoint ID where the higher price is observed.
    asset:
        Asset symbol (e.g. ``"ETH"``, ``"USDC"``).
    src_price:
        Price on the source chain (in quote-token units, 18-decimal fixed).
    dst_price:
        Price on the destination chain.
    spread_bps:
        Spread in basis points (``(dst_price - src_price) / src_price * 10000``).
    detected_at:
        Unix timestamp when the opportunity was detected.
    lz_guid:
        GUID of the LayerZero packet that carried this price update.
    """
    opportunity_id: str   = field(default_factory=lambda: str(uuid.uuid4()))
    src_eid:        int   = 0
    dst_eid:        int   = 0
    asset:          str   = ""
    src_price:      float = 0.0
    dst_price:      float = 0.0
    spread_bps:     float = 0.0
    detected_at:    int   = field(default_factory=lambda: int(time.time()))
    lz_guid:        str   = ""


class ArbitrageChannel:
    """Cross-chain arbitrage price-feed and opportunity relay.

    Wraps a :class:`LayerZeroAdapter` to broadcast price updates across
    chains and detect arbitrage windows above a configurable spread
    threshold.

    The channel models the Generative Augmented Structure (GAS) embedding
    flow: each received price update is treated as a 32-bit float32 feature
    vector that the CROF Kernel32 VM can consume for signal scoring.

    Parameters
    ----------
    lz:
        The underlying :class:`LayerZeroAdapter` used for cross-chain
        messaging.
    asset:
        The asset being monitored (e.g. ``"ETH/USDC"``).
    min_spread_bps:
        Minimum spread in basis points required to record an opportunity
        (default 10 bps = 0.10 %).
    """

    def __init__(
        self,
        lz: LayerZeroAdapter,
        asset: str = "ETH/USDC",
        min_spread_bps: float = 10.0,
    ) -> None:
        self._lz              = lz
        self.asset            = asset
        self.min_spread_bps   = min_spread_bps
        self._src_price:  float | None = None
        self._dst_price:  float | None = None
        self._opportunities:  list[ArbitrageOpportunity] = []

    @property
    def opportunities(self) -> list[ArbitrageOpportunity]:
        """Recorded arbitrage opportunities above the spread threshold."""
        return list(self._opportunities)

    def publish_price(
        self,
        price: float,
        domain_id: str = "arb-src",
        *,
        is_source: bool = True,
    ) -> DispatchResult:
        """Broadcast a price update via LayerZero and check for arbitrage.

        The price is JSON-encoded and wrapped in a CROF :class:`Envelope`
        before being sent through the :class:`LayerZeroAdapter`.

        Parameters
        ----------
        price:
            Asset price in the quote token.
        domain_id:
            CROF domain identifier for the envelope.
        is_source:
            When ``True`` (default), update the source-chain price.
            When ``False``, update the destination-chain price.
        """
        if is_source:
            self._src_price = price
        else:
            self._dst_price = price

        payload = json.dumps({
            "asset":    self.asset,
            "price":    price,
            "side":     "src" if is_source else "dst",
            "eid":      self._lz.src_eid if is_source else self._lz.dst_eid,
        }).encode()
        env = Envelope.build(
            actor_id   = "arb-channel",
            domain_id  = domain_id,
            payload    = payload,
            payload_type = "application/json",
            operation_type = "Event",
        )
        result = self._lz.dispatch(env)
        self._check_opportunity(result.metadata.get("lz_guid", ""))
        return result

    def _check_opportunity(self, lz_guid: str) -> None:
        if self._src_price is None or self._dst_price is None:
            return
        if self._src_price <= 0:
            return
        spread_bps = (
            (self._dst_price - self._src_price) / self._src_price * 10_000
        )
        if abs(spread_bps) >= self.min_spread_bps:
            opp = ArbitrageOpportunity(
                src_eid    = self._lz.src_eid,
                dst_eid    = self._lz.dst_eid,
                asset      = self.asset,
                src_price  = self._src_price,
                dst_price  = self._dst_price,
                spread_bps = spread_bps,
                lz_guid    = lz_guid,
            )
            self._opportunities.append(opp)

    def as_float32_feature(self) -> bytes:
        """Encode the latest price spread as a 4-byte float32 (GAS embedding unit).

        Returns ``b'\\x00\\x00\\x00\\x00'`` when insufficient price data is
        available.  This value can be loaded directly into the Kernel32 VM
        (``EMB_STORE`` opcode) as a 32-bit embedding scalar.
        """
        if self._src_price is None or self._dst_price is None:
            return struct.pack("f", 0.0)
        spread = abs(self._dst_price - self._src_price) / max(self._src_price, 1e-12)
        return struct.pack("f", float(spread))

    def __repr__(self) -> str:
        return (
            f"ArbitrageChannel(asset={self.asset!r}, "
            f"opportunities={len(self._opportunities)}, "
            f"spread_threshold={self.min_spread_bps}bps)"
        )
