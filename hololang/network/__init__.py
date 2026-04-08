"""hololang.network package – gRPC, WebSocket, webhook, and REST API."""
from hololang.network.api import Channel, Protocol, Message, ApiEndpoint, ApiRoute
from hololang.network.websocket import WebSocketConnection, WebSocketServer
from hololang.network.webhook import Webhook, GrpcChannel, WebhookEvent

__all__ = [
    "Channel", "Protocol", "Message", "ApiEndpoint", "ApiRoute",
    "WebSocketConnection", "WebSocketServer",
    "Webhook", "GrpcChannel", "WebhookEvent",
]
