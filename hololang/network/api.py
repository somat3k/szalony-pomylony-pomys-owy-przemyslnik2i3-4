"""Network API layer – protocol definitions, channel registry, and
REST/WebSocket/gRPC endpoint management for HoloLang programs.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Protocol enum
# ---------------------------------------------------------------------------

class Protocol(Enum):
    HTTP    = "http"
    HTTPS   = "https"
    WS      = "ws"
    WSS     = "wss"
    GRPC    = "grpc"
    GRPCS   = "grpcs"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message:
    """Protocol-agnostic message envelope."""

    def __init__(
        self,
        topic: str,
        payload: Any,
        msg_id: str = "",
        timestamp: float | None = None,
    ) -> None:
        self.msg_id    = msg_id or str(uuid.uuid4())
        self.topic     = topic
        self.payload   = payload
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "msg_id":    self.msg_id,
            "topic":     self.topic,
            "payload":   self.payload,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(d["topic"], d["payload"], d.get("msg_id", ""), d.get("timestamp"))

    def __repr__(self) -> str:
        return f"Message(topic={self.topic!r}, payload={self.payload!r})"


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class Channel:
    """A named communication channel with a specific protocol.

    Parameters
    ----------
    name:
        Channel identifier.
    protocol:
        Protocol string (``"grpc"``, ``"ws"``, ``"http"``, …).
    host:
        Remote host (default ``"localhost"``).
    port:
        Remote port.
    """

    def __init__(
        self,
        name: str,
        protocol: str = "grpc",
        host: str = "localhost",
        port: int | None = None,
    ) -> None:
        self.name      = name
        self.protocol  = Protocol(protocol) if protocol in Protocol._value2member_map_ else Protocol.HTTP
        self.host      = host
        self.port      = port or self._default_port()
        self._handlers: list[Callable[[Message], Any]] = []
        self._inbox:    list[Message] = []
        self._lock = threading.Lock()
        self._open = False

    def _default_port(self) -> int:
        defaults = {
            Protocol.HTTP:    80,
            Protocol.HTTPS:   443,
            Protocol.WS:      8080,
            Protocol.WSS:     8443,
            Protocol.GRPC:    50051,
            Protocol.GRPCS:   50052,
            Protocol.WEBHOOK: 9000,
        }
        return defaults.get(self.protocol, 8080)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def emit(self, payload: Any, topic: str = "default") -> Message:
        msg = Message(topic=topic, payload=payload)
        with self._lock:
            self._inbox.append(msg)
        for handler in self._handlers:
            handler(msg)
        return msg

    def listen(self, handler: Callable[[Message], Any]) -> None:
        self._handlers.append(handler)

    def receive_all(self) -> list[Message]:
        with self._lock:
            msgs = list(self._inbox)
            self._inbox.clear()
        return msgs

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"{self.protocol.value}://{self.host}:{self.port}"

    def __repr__(self) -> str:
        return (
            f"Channel(name={self.name!r}, protocol={self.protocol.value}, "
            f"url={self.url}, open={self._open})"
        )


# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------

class HttpMethod(Enum):
    GET    = "GET"
    POST   = "POST"
    PUT    = "PUT"
    DELETE = "DELETE"
    PATCH  = "PATCH"


class ApiRoute:
    """A single API route handler."""

    def __init__(
        self,
        path: str,
        method: str = "GET",
        handler: Callable | None = None,
    ) -> None:
        self.path    = path
        self.method  = HttpMethod(method.upper())
        self.handler = handler
        self._calls: list[dict] = []

    def call(self, body: Any = None, params: dict | None = None) -> Any:
        entry = {"body": body, "params": params or {}, "time": time.time()}
        self._calls.append(entry)
        if self.handler:
            return self.handler(body, params or {})
        return {"status": "ok", "route": self.path}

    def __repr__(self) -> str:
        return f"ApiRoute({self.method.value} {self.path})"


class ApiEndpoint:
    """Named REST API endpoint collection."""

    def __init__(self, name: str, base_path: str = "/api/v1") -> None:
        self.name      = name
        self.base_path = base_path
        self._routes:  dict[str, ApiRoute] = {}
        self.port:     int = 8000

    def route(
        self,
        path: str,
        method: str = "GET",
        handler: Callable | None = None,
    ) -> ApiRoute:
        key = f"{method.upper()} {path}"
        r = ApiRoute(path=self.base_path + path, method=method, handler=handler)
        self._routes[key] = r
        return r

    def get(self, path: str, handler: Callable | None = None) -> ApiRoute:
        return self.route(path, "GET", handler)

    def post(self, path: str, handler: Callable | None = None) -> ApiRoute:
        return self.route(path, "POST", handler)

    def call(self, method: str, path: str, body: Any = None) -> Any:
        key = f"{method.upper()} {path}"
        r = self._routes.get(key)
        if r is None:
            return {"error": f"Route {key!r} not found"}
        return r.call(body)

    def __repr__(self) -> str:
        return (
            f"ApiEndpoint(name={self.name!r}, "
            f"routes={len(self._routes)}, port={self.port})"
        )
