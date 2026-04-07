"""HoloRuntime – top-level orchestrator for the HoloLang platform.

Wires together:
* Language front-end (lexer → parser → interpreter)
* Tensor processing (pool, graph, safetensor)
* Device layer (laser, mirrors, sensors)
* Mesh canvas MDI
* Virtual machine (kernel pool, block controller)
* Network layer (channels, webhooks, gRPC)
* Documentation system (sessions, skills, directories)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hololang.lang.interpreter import Interpreter, Environment
from hololang.lang.parser import parse
from hololang.tensor.pool import TensorPool
from hololang.tensor.tensor import Tensor
from hololang.device.holographic import LaserDevice, GalvanizedMirror, Sensor, SensorType
from hololang.mesh.canvas import Canvas
from hololang.mesh.display import Display
from hololang.vm.runtime import PoolRuntime
from hololang.vm.controller import BlockController
from hololang.network.api import Channel, ApiEndpoint
from hololang.network.webhook import Webhook, GrpcChannel
from hololang.network.websocket import WebSocketServer
from hololang.docs.session import Session
from hololang.docs.skills import get_registry
from hololang.docs.directory import DocDirectory


class HoloRuntime:
    """Central orchestrator for a HoloLang program execution environment.

    Parameters
    ----------
    session_name:
        Name for the new session.
    output_hook:
        Callable for interpreter output; defaults to ``print``.

    Example::

        rt = HoloRuntime()
        rt.run_string('''
            device Projector {
                type: "galvanized-laser"
                channels: 3
            }
            debug Projector
        ''')
    """

    def __init__(
        self,
        session_name: str = "default",
        output_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.session = Session(name=session_name)
        self.session.log("runtime_init")

        # Core subsystems
        self.tensor_pool = TensorPool(name="main_pool")
        self.canvas      = Canvas(name="main_canvas")
        self.vm          = PoolRuntime(name="main_vm")
        self.controller  = BlockController(name="main_controller")
        self.vm.set_controller(self.controller)

        # Network
        self.ws_server   = WebSocketServer()
        self._channels:  dict[str, Channel] = {}
        self._webhooks:  dict[str, Webhook] = {}
        self._grpc_chans: dict[str, GrpcChannel] = {}

        # Devices
        self._devices:   dict[str, Any] = {}

        # Docs / skills
        self.skill_registry = get_registry()
        self.doc_root = DocDirectory("hololang_docs")

        # Interpreter
        self._output: list[str] = []
        self._hook = output_hook or self._default_output
        env = Environment()
        self._env = env
        self._interpreter = Interpreter(env=env, output_hook=self._hook)
        self._setup_runtime_builtins()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _default_output(self, text: str) -> None:
        self._output.append(text)
        print(text)

    def get_output(self) -> list[str]:
        return list(self._output)

    # ------------------------------------------------------------------
    # Runtime builtins exposed to HoloLang programs
    # ------------------------------------------------------------------

    def _setup_runtime_builtins(self) -> None:
        env = self._env

        # Tensor pool access
        env.define("tensor_pool", self.tensor_pool)
        env.define("alloc", lambda tag, *dims: self.tensor_pool.allocate(tag, *dims))

        # Canvas access
        env.define("main_canvas", self.canvas)

        # Device factories
        env.define(
            "make_laser",
            lambda did, name="Laser": self._register_device(
                did, LaserDevice(did, name)
            ),
        )
        env.define(
            "make_mirror",
            lambda did, name="Mirror": self._register_device(
                did, GalvanizedMirror(did, name)
            ),
        )
        env.define(
            "make_sensor",
            lambda did, name="Sensor": self._register_device(
                did, Sensor(did, name, SensorType.PHOTODIODE)
            ),
        )
        env.define("get_device", lambda did: self._devices.get(did))

        # VM access
        env.define("vm", self.vm)
        env.define("controller", self.controller)

        # Network
        env.define("open_channel", self._open_channel)
        env.define("add_webhook", self._add_webhook)

        # Docs
        env.define("session", self.session)
        env.define("doc_root", self.doc_root)
        env.define("skills", self.skill_registry)

    def _register_device(self, did: str, device: Any) -> Any:
        self._devices[did] = device
        device.initialise()
        return device

    def _open_channel(self, name: str, protocol: str = "grpc") -> Channel:
        ch = Channel(name=name, protocol=protocol)
        ch.open()
        self._channels[name] = ch
        return ch

    def _add_webhook(self, name: str, url: str = "") -> Webhook:
        wh = Webhook(name=name, url=url)
        self._webhooks[name] = wh
        return wh

    # ------------------------------------------------------------------
    # Program execution
    # ------------------------------------------------------------------

    def run_string(self, source: str) -> None:
        """Parse and execute a HoloLang source string."""
        self.session.log("run_string")
        try:
            program = parse(source)
            self._interpreter.eval_program(program)
        except Exception as exc:
            self.session.log("error", str(exc))
            raise

    def run_file(self, path: str) -> None:
        """Load and execute a ``.hl`` file."""
        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"HoloLang file not found: {path}")
        source = fpath.read_text(encoding="utf-8")
        self.session.log("run_file", path)
        self.run_string(source)

    # ------------------------------------------------------------------
    # Introspection / REPL helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Look up a variable in the runtime environment."""
        try:
            return self._env.lookup(name)
        except NameError:
            return None

    def set(self, name: str, value: Any) -> None:
        self._env.define(name, value)

    def display_canvas(self, use_color: bool = False) -> str:
        disp = Display(self.canvas, use_color=use_color)
        return disp.render_terminal()

    # ------------------------------------------------------------------
    # Session / lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.session.end()
        self.tensor_pool.close()
        self.vm.close()

    def __enter__(self) -> "HoloRuntime":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"HoloRuntime(session={self.session.name!r}, "
            f"devices={len(self._devices)}, "
            f"tiles={len(self.canvas)})"
        )
