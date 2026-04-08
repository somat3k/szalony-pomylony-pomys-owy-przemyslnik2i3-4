"""Webhook dispatcher for HoloLang network integrations.

Dispatches HTTP POST payloads to registered callback URLs (simulation)
and exposes a listener interface for incoming webhooks.
"""

from __future__ import annotations

import json
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Webhook event
# ---------------------------------------------------------------------------

@dataclass
class WebhookEvent:
    """An incoming or outgoing webhook payload."""
    event_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event:     str = ""
    payload:   Any = None
    timestamp: float = field(default_factory=time.time)
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id":   self.event_id,
            "event":      self.event,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
            "source_url": self.source_url,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

class Webhook:
    """Named webhook registration – simulates HTTP POST dispatch.

    Parameters
    ----------
    name:
        Human-readable label.
    url:
        Target URL for outgoing webhooks.
    secret:
        Optional shared secret for HMAC signature (not enforced in sim).
    """

    def __init__(
        self,
        name: str,
        url: str = "",
        secret: str = "",
    ) -> None:
        self.name   = name
        self.url    = url
        self.secret = secret
        self._handlers:  dict[str, list[Callable[[WebhookEvent], Any]]] = {}
        self._sent:      list[WebhookEvent] = []
        self._received:  list[WebhookEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Outgoing dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        event: str,
        payload: Any,
        target_url: str = "",
    ) -> WebhookEvent:
        """Simulate dispatching a webhook to *target_url*."""
        ev = WebhookEvent(
            event=event,
            payload=payload,
            source_url=self.url,
        )
        with self._lock:
            self._sent.append(ev)
        # In a real implementation this would perform an HTTP POST
        return ev

    # ------------------------------------------------------------------
    # Incoming listener
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable[[WebhookEvent], Any]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def receive(self, event: str, payload: Any, source_url: str = "") -> WebhookEvent:
        """Simulate receiving an incoming webhook."""
        ev = WebhookEvent(event=event, payload=payload, source_url=source_url)
        with self._lock:
            self._received.append(ev)
        for handler in self._handlers.get(event, []):
            handler(ev)
        for handler in self._handlers.get("*", []):
            handler(ev)
        return ev

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def sent_events(self) -> list[WebhookEvent]:
        with self._lock:
            return list(self._sent)

    def received_events(self) -> list[WebhookEvent]:
        with self._lock:
            return list(self._received)

    def __repr__(self) -> str:
        return (
            f"Webhook(name={self.name!r}, url={self.url!r}, "
            f"sent={len(self._sent)}, received={len(self._received)})"
        )


# ---------------------------------------------------------------------------
# gRPC channel stub
# ---------------------------------------------------------------------------

class GrpcChannel:
    """Lightweight gRPC channel stub for HoloLang programs.

    In production this wraps ``grpc.insecure_channel`` / ``grpc.secure_channel``.
    For testing / simulation it implements the same interface without an
    actual network dependency.
    """

    def __init__(self, target: str, port: int = 50051, secure: bool = False) -> None:
        self.target = target
        self.port   = port
        self.secure = secure
        self._open  = False
        self._calls: list[dict] = []

    @property
    def address(self) -> str:
        return f"{self.target}:{self.port}"

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def call(self, method: str, request: Any) -> Any:
        """Simulate a unary gRPC call."""
        if not self._open:
            raise ConnectionError(f"gRPC channel to {self.address} is not open")
        record = {
            "method":    method,
            "request":   request,
            "timestamp": time.time(),
            "response":  {"status": "ok", "method": method},
        }
        self._calls.append(record)
        return record["response"]

    def call_history(self) -> list[dict]:
        return list(self._calls)

    def __repr__(self) -> str:
        return (
            f"GrpcChannel(address={self.address!r}, "
            f"secure={self.secure}, open={self._open})"
        )
