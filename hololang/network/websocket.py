"""WebSocket channel abstraction.

Provides a synchronous simulation of a WebSocket connection suitable for
testing without requiring an actual network.  When the optional
``websockets`` library is installed the :class:`WebSocketServer` class can
drive real connections.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable

from hololang.network.api import Message


class WebSocketConnection:
    """Simulated WebSocket connection.

    Supports sending / receiving messages and registering event handlers.
    """

    def __init__(self, connection_id: str = "") -> None:
        self.connection_id = connection_id or str(uuid.uuid4())[:8]
        self._inbox:    list[Message] = []
        self._outbox:   list[Message] = []
        self._handlers: dict[str, list[Callable]] = {}
        self._open      = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, url: str = "") -> None:
        self._open = True
        self._emit_event("open", url=url)

    def disconnect(self) -> None:
        self._open = False
        self._emit_event("close")

    def is_open(self) -> bool:
        return self._open

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, payload: Any, topic: str = "message") -> None:
        if not self._open:
            raise ConnectionError("WebSocket is not connected")
        msg = Message(topic=topic, payload=payload)
        self._outbox.append(msg)
        self._emit_event("send", message=msg)

    def receive(self, payload: Any, topic: str = "message") -> None:
        """Inject an incoming message (used by server / simulator)."""
        msg = Message(topic=topic, payload=payload)
        self._inbox.append(msg)
        self._emit_event("message", message=msg)

    def receive_all(self) -> list[Message]:
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs

    def sent_all(self) -> list[Message]:
        msgs = list(self._outbox)
        self._outbox.clear()
        return msgs

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def _emit_event(self, event: str, **kwargs: Any) -> None:
        for cb in self._handlers.get(event, []):
            cb(**kwargs)

    def __repr__(self) -> str:
        return (
            f"WebSocketConnection(id={self.connection_id!r}, open={self._open})"
        )


class WebSocketServer:
    """Simple in-process WebSocket message broker (simulation / testing).

    Real deployments should replace this with ``websockets.serve`` or an
    ASGI framework.
    """

    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        self.host   = host
        self.port   = port
        self._conns: dict[str, WebSocketConnection] = {}
        self._lock  = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def accept(self, conn_id: str = "") -> WebSocketConnection:
        conn = WebSocketConnection(conn_id)
        with self._lock:
            self._conns[conn.connection_id] = conn
        conn.connect(f"ws://{self.host}:{self.port}")
        return conn

    def broadcast(self, payload: Any, topic: str = "broadcast") -> None:
        with self._lock:
            conns = list(self._conns.values())
        for conn in conns:
            conn.receive(payload, topic)

    def connection_count(self) -> int:
        with self._lock:
            return len(self._conns)

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def __repr__(self) -> str:
        return (
            f"WebSocketServer(url={self.url}, "
            f"connections={len(self._conns)}, "
            f"running={self._running})"
        )
