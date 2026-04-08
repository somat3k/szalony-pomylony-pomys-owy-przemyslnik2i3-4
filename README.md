# HoloLang

> **Custom DSL for holographic device control, tensor processing, and light-manipulation automation.**

HoloLang is a beginner-friendly programming language and runtime platform built for:

- **Holographic device control** – lasers, galvanised mirrors, sensors
- **Light manipulation** – beam spectrum, wavelength, scan patterns
- **Tensor processing** – N-dimensional tensors, SafeTensor bounds-checking, computation graphs, pooled runtimes
- **Mesh MDI canvas** – tile-based multi-document interface with impulse cycle propagation
- **Replicable VM** – stack-based kernel VM, block controller, generative pipeline
- **Network stack** – gRPC channel, WebSocket, webhook, REST API (protocol-agnostic)
- **Documentation system** – sessions, skill registry, hierarchical doc directories

---

## Quick Start

```bash
pip install -e .
hololang info          # show version and subsystem overview
hololang skills        # list all skills
hololang run examples/hello_holo.hl
hololang repl          # interactive REPL
```

---

## Language Overview

HoloLang (`.hl` files) has a clean, readable syntax inspired by Rust, Kotlin, and domain-specific notation for photonics engineering.

### Devices

```hololang
device GreenLaser {
    type:          "solid-state-laser"
    wavelength_nm: 532
    max_power_mw:  150
}

device MirrorH {
    type:    "galvanized-mirror"
    axis:    "horizontal"
    min_deg: -30
    max_deg:  30
}
```

### Tensors & SafeTensors

```hololang
tensor FrameBuffer[1080][1920] {
    dtype:   "float32"
    min_val: 0.0
    max_val: 1.0
    clamp:   true
    pool:    "replicated"
}
```

### Enums

```hololang
enum ScanDirection {
    LEFT_RIGHT    = 0,
    BOUSTROPHEDON = 2
}

let mode = ScanDirection.BOUSTROPHEDON
```

### Functions

```hololang
fn normalize(value, min_v, max_v) {
    return (value - min_v) / (max_v - min_v)
}

let r = normalize(0.7, 0.0, 1.0)
```

### Mesh Canvas & Impulse Cycles

```hololang
mesh HoloMesh(FrameBuffer) {
    tile(0, 0) -> GreenLaser
    tile(0, 1) -> MirrorH
    tile(1, 0) -> BeamSensor
}

impulse(tile(0, 0), tile(0, 1)) -> GreenLaser
invoke(HoloMesh)
```

### Kernel VM & Pool Runtime

```hololang
kernel ScanKernel {
    replicas:   4
    max_cycles: 50000
}

pool ScanPool {
    kernels: ScanKernel
    workers: 4
}
```

### Network Stack

```hololang
channel DataChannel(grpc) {
    host: "localhost"
    port: 50051
}

webhook StatusHook {
    url:    "http://localhost:9000/events"
    events: "all"
}

api HoloAPI {
    base_path: "/api/v1"
    port:      8000
}

emit DataChannel -> "beam_state_update"
invoke(HoloMesh) -> DataChannel -> HoloAPI
```

### Sessions & Skills

```hololang
@session("my-project")

session MySession {
    skill: "laser_control"
    skill: "tensor_pool"
    skill: "mesh_canvas"
}
```

---

## Project Structure

```
hololang/
├── lang/           # Language front-end
│   ├── lexer.py    # Tokenizer
│   ├── ast_nodes.py# AST node definitions
│   ├── parser.py   # Recursive-descent parser
│   └── interpreter.py  # Tree-walk interpreter
├── tensor/         # Tensor subsystem
│   ├── tensor.py   # N-dimensional Tensor
│   ├── safetensor.py   # Bounds-checked SafeTensor
│   ├── graph.py    # Computation graph (matmul, relu, normalize…)
│   └── pool.py     # Pooled tensor runtime
├── device/         # Device layer
│   └── holographic.py  # LaserDevice, GalvanizedMirror, Sensor
├── mesh/           # MDI canvas
│   ├── tile.py     # Mesh tile with impulse connections
│   ├── canvas.py   # Sparse 2-D tile canvas
│   └── display.py  # Terminal / SVG / JSON rendering
├── vm/             # Virtual machine
│   ├── kernel.py   # Stack-based kernel + instruction set
│   ├── controller.py   # Generative block controller
│   └── runtime.py  # Pool runtime with replicable kernels
├── network/        # Network layer
│   ├── api.py      # Channel, Message, ApiEndpoint
│   ├── websocket.py    # WebSocketConnection, WebSocketServer
│   └── webhook.py  # Webhook, GrpcChannel, WebhookEvent
├── docs/           # Documentation system
│   ├── session.py  # Session lifecycle & artefacts
│   ├── skills.py   # Skill registry
│   └── directory.py    # Hierarchical DocDirectory
├── runtime.py      # HoloRuntime orchestrator
└── cli.py          # Command-line interface

examples/
├── hello_holo.hl       # Hello, World!
├── laser_config.hl     # Laser + mirror + sensor wiring
├── tensor_graph.hl     # Computation graph pipeline
└── full_system.hl      # Complete system demo

tests/
├── test_lang_lexer.py
├── test_lang_interpreter.py
├── test_tensor.py
├── test_device.py
├── test_mesh.py
├── test_vm.py
├── test_network.py
├── test_docs.py
└── test_runtime.py
```

---

## Running Examples

```bash
hololang run examples/hello_holo.hl
hololang run examples/laser_config.hl
hololang run examples/tensor_graph.hl
hololang run examples/full_system.hl

# Parse check only
hololang check examples/full_system.hl

# Render canvas after execution
hololang canvas examples/hello_holo.hl --no-color
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## CLI Reference

| Command | Description |
|---|---|
| `hololang run FILE` | Execute a `.hl` program |
| `hololang check FILE` | Parse-check a `.hl` file |
| `hololang repl` | Interactive REPL |
| `hololang info` | Show version and subsystem list |
| `hololang skills` | List all registered skills |
| `hololang canvas FILE` | Run program and render its mesh canvas |
