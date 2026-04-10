"""Plane F — Interdomain Plane: pluggable domain adapters.

The CROF fabric communicates across domain boundaries through a unified
:class:`DomainAdapter` interface.  Concrete adapters are provided for:

* :class:`GrpcAdapter`       — direct gRPC service-to-service traffic
* :class:`MqttAdapter`       — device / embedded-network messaging
* :class:`LayerZeroAdapter`  — omnichain trust-minimised bridge (LayerZero V2)

LayerZero integration
---------------------
`LayerZero <https://layerzero.network>`_ is a cross-chain messaging protocol.
The :class:`LayerZeroAdapter` follows the **LayerZero V2 OApp** standard:

* Messages are packaged as :class:`LzPacket` instances carrying a
  ``src_eid`` (source endpoint ID), ``dst_eid`` (destination endpoint ID),
  ``guid``, ``nonce``, and ``message`` bytes — matching the V2
  ``MessagingFee`` / ``Origin`` / ``SendParam`` schema.
* In production the adapter would call the LayerZero Endpoint contract via
  ``web3.py`` or the LayerZero SDK.  Here it provides the full protocol
  logic and exposes hook points for wiring to a live network.
* DVN (Decentralised Verification Network) receipt simulation is included
  via :meth:`LayerZeroAdapter.simulate_dvn_receive`.

Each adapter returns a :class:`DispatchResult` that records the delivery
status, latency, and any error messages.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
        """Send *envelope* to the remote domain.

        Returns
        -------
        DispatchResult
        """

    @abstractmethod
    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        """Query state from the remote domain.

        Parameters
        ----------
        query:
            Adapter-specific query parameters.

        Returns
        -------
        dict
            The remote state response.
        """


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
    """

    def __init__(
        self,
        endpoint: str,
        service: str = "InteropService",
        timeout_ms: int = 5000,
    ) -> None:
        self.endpoint   = endpoint
        self.service    = service
        self.timeout_ms = timeout_ms
        self._call_hook: Callable[[str, bytes], bytes] | None = None

    def register_call_hook(self, hook: Callable[[str, bytes], bytes]) -> None:
        """Register a callback that stands in for the real gRPC channel."""
        self._call_hook = hook

    @property
    def adapter_type(self) -> str:
        return "grpc"

    def dispatch(self, envelope: Envelope) -> DispatchResult:
        t0      = time.perf_counter()
        payload = envelope.to_json().encode()
        error   = ""
        if self._call_hook is not None:
            try:
                self._call_hook(f"{self.service}.DispatchInterdomain", payload)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {"endpoint": self.endpoint, "service": self.service},
        )

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        if self._call_hook is not None:
            try:
                raw = self._call_hook(
                    f"{self.service}.QueryRemoteState",
                    json.dumps(query).encode(),
                )
                return json.loads(raw)
            except Exception:  # noqa: BLE001
                pass
        return {"status": "simulated", "endpoint": self.endpoint}


# ---------------------------------------------------------------------------
# MQTT adapter
# ---------------------------------------------------------------------------

class MqttAdapter(DomainAdapter):
    """Device / embedded-network adapter using the MQTT pub/sub model.

    Parameters
    ----------
    broker:
        MQTT broker address in ``"host:port"`` format.
    topic_prefix:
        Prefix for all published topics (e.g. ``"crof/domain/edge"``).
    qos:
        MQTT QoS level (0, 1, or 2).
    """

    def __init__(
        self,
        broker: str,
        topic_prefix: str = "crof",
        qos: int = 1,
    ) -> None:
        self.broker       = broker
        self.topic_prefix = topic_prefix
        self.qos          = qos
        self._publish_hook: Callable[[str, bytes, int], None] | None = None

    def register_publish_hook(
        self, hook: Callable[[str, bytes, int], None]
    ) -> None:
        """Register a callback that stands in for the real MQTT client."""
        self._publish_hook = hook

    @property
    def adapter_type(self) -> str:
        return "mqtt"

    def dispatch(self, envelope: Envelope) -> DispatchResult:
        topic   = f"{self.topic_prefix}/{envelope.domain_id}/{envelope.operation_type}"
        payload = envelope.to_json().encode()
        t0      = time.perf_counter()
        error   = ""
        if self._publish_hook is not None:
            try:
                self._publish_hook(topic, payload, self.qos)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {"broker": self.broker, "topic": topic, "qos": self.qos},
        )

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        return {"status": "simulated", "broker": self.broker}


# ---------------------------------------------------------------------------
# LayerZero V2 data structures
# ---------------------------------------------------------------------------

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
        Encoded execution options (gas limit, native drop, etc.).
    """
    src_eid:  int   = 0
    dst_eid:  int   = 0
    sender:   str   = ""
    receiver: str   = ""
    nonce:    int   = 0
    guid:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    message:  bytes = b""
    options:  bytes = b""

    def encode(self) -> bytes:
        """Return a canonical binary encoding of the packet header."""
        header = json.dumps({
            "src_eid": self.src_eid,
            "dst_eid": self.dst_eid,
            "sender":  self.sender,
            "receiver": self.receiver,
            "nonce":   self.nonce,
        }, sort_keys=True, separators=(",", ":")).encode()
        return header + self.message

    @property
    def keccak_guid(self) -> str:
        """Return the keccak-256 GUID of the packet header (SHA-256 here)."""
        return hashlib.sha256(self.encode()).hexdigest()


@dataclass
class DvnVerification:
    """Simulated DVN (Decentralised Verification Network) receipt."""
    packet_guid:  str
    dvn_id:       str
    verified:     bool
    timestamp:    int = field(default_factory=lambda: int(time.time()))
    proof:        bytes = b""


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
    """

    def __init__(
        self,
        src_eid:      int = 30101,
        dst_eid:      int = 30110,
        sender:       str = "0x" + "00" * 20,
        receiver:     str = "0x" + "00" * 20,
        endpoint_url: str = "",
        dvn_count:    int = 1,
        send_hook:    Callable[["LzPacket"], None] | None = None,
    ) -> None:
        self.src_eid      = src_eid
        self.dst_eid      = dst_eid
        self.sender       = sender
        self.receiver     = receiver
        self.endpoint_url = endpoint_url
        self.dvn_count    = dvn_count
        self._send_hook   = send_hook
        self._nonce:      int = 0
        self._sent:       list[LzPacket] = []
        self._dvn_receipts: list[DvnVerification] = []

    @property
    def adapter_type(self) -> str:
        return "layerzero"

    # ------------------------------------------------------------------
    # OApp _lzSend equivalent
    # ------------------------------------------------------------------

    def _build_packet(self, envelope: Envelope) -> LzPacket:
        self._nonce += 1
        msg   = envelope.to_json().encode()
        pkt   = LzPacket(
            src_eid  = self.src_eid,
            dst_eid  = self.dst_eid,
            sender   = self.sender,
            receiver = self.receiver,
            nonce    = self._nonce,
            message  = msg,
        )
        pkt.guid = pkt.keccak_guid
        return pkt

    def dispatch(self, envelope: Envelope) -> DispatchResult:
        """Package *envelope* as an LzPacket and dispatch it.

        In production, this calls the LayerZero Endpoint's ``send()`` function.
        Here it invokes the optional ``send_hook`` and records the packet.
        """
        t0  = time.perf_counter()
        pkt = self._build_packet(envelope)
        error = ""
        try:
            if self._send_hook is not None:
                self._send_hook(pkt)
            self._sent.append(pkt)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DispatchResult(
            packet_id    = pkt.guid,
            adapter_type = self.adapter_type,
            status       = "failed" if error else "dispatched",
            latency_ms   = dt_ms,
            error        = error,
            metadata     = {
                "src_eid":  self.src_eid,
                "dst_eid":  self.dst_eid,
                "nonce":    pkt.nonce,
                "lz_guid":  pkt.guid,
            },
        )

    # ------------------------------------------------------------------
    # DVN simulation (_lzReceive path)
    # ------------------------------------------------------------------

    def simulate_dvn_receive(self, packet: LzPacket) -> list[DvnVerification]:
        """Simulate DVN verification receipts for a delivered packet.

        In production, the LayerZero DVN network submits ``assignJob`` /
        ``verifyAndAttest`` transactions to the receiving chain's Endpoint.
        This method generates *dvn_count* synthetic verification receipts.

        Returns
        -------
        list[DvnVerification]
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

    def query_remote_state(self, query: dict[str, Any]) -> dict[str, Any]:
        """Simulate a LayerZero ``lzRead`` remote state query."""
        return {
            "status":       "simulated",
            "src_eid":      self.src_eid,
            "dst_eid":      self.dst_eid,
            "packets_sent": len(self._sent),
            "dvn_verified": len(self._dvn_receipts),
            "query":        query,
        }

    @property
    def sent_packets(self) -> list[LzPacket]:
        return list(self._sent)

    @property
    def dvn_receipts(self) -> list[DvnVerification]:
        return list(self._dvn_receipts)

    def __repr__(self) -> str:
        return (
            f"LayerZeroAdapter(src_eid={self.src_eid}, dst_eid={self.dst_eid}, "
            f"sent={len(self._sent)}, dvn={len(self._dvn_receipts)})"
        )
