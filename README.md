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
├── lang/               # Language front-end
│   ├── lexer.py        # Tokenizer (90+ token types, multi-style comments)
│   ├── ast_nodes.py    # 50 AST node definitions
│   ├── parser.py       # Recursive-descent parser
│   └── interpreter.py  # Tree-walk interpreter
├── tensor/             # Tensor subsystem
│   ├── helpers.py      # Row-major strides, flat-index, element-count utilities
│   ├── tensor.py       # N-dimensional Tensor (pure Python, NumPy-optional)
│   ├── safetensor.py   # Bounds-checked SafeTensor with CRC32 serialization
│   ├── ops.py          # Activation & transform functions (relu, sigmoid, softmax…)
│   ├── matrix.py       # Threaded MatrixEngine + ParameterMultiplier scheduler
│   ├── transforms.py   # TransformChain (sequential) + ParallelTransformGroup
│   ├── hyperparams.py  # HyperParameter + HyperParamSpace with scheduling
│   ├── batch.py        # SpreadTensor shards + BatchContainer parallel apply
│   ├── graph_node.py   # GraphNode (standalone DAG node)
│   ├── graph.py        # ComputationGraph (matmul, relu, normalize, forward pass)
│   └── pool.py         # TensorPool – named tensor registry with map/reduce
├── device/             # Device layer
│   └── holographic.py  # LaserDevice, GalvanizedMirror, Sensor (simulation)
├── mesh/               # MDI canvas
│   ├── tile.py         # Mesh tile with impulse connections
│   ├── canvas.py       # Sparse 2-D tile canvas with impulse cycle
│   └── display.py      # Terminal (ANSI) / SVG / JSON rendering
├── vm/                 # Virtual machine
│   ├── state.py        # KernelState enum (IDLE, RUNNING, SUSPENDED, FINISHED, ERROR)
│   ├── instruction.py  # Instruction dataclass (opcode + operands)
│   ├── block.py        # Block dataclass (named callable unit)
│   ├── kernel.py       # Stack-based kernel with 26-opcode ISA + replication
│   ├── controller.py   # BlockController – DAG-based generative pipeline
│   └── runtime.py      # PoolRuntime – kernel pool with optional parallelism
├── network/            # Network layer
│   ├── api.py          # Channel, Message, ApiRoute, ApiEndpoint
│   ├── websocket.py    # WebSocketConnection, WebSocketServer (in-process)
│   └── webhook.py      # Webhook, GrpcChannel, WebhookEvent (in-process)
├── docs/               # Documentation system
│   ├── session.py      # Session lifecycle, artefacts & event log
│   ├── skills.py       # SkillRegistry with 24 built-in skills
│   └── directory.py    # Hierarchical DocDirectory with full-text search
├── runtime.py          # HoloRuntime orchestrator (wires all subsystems)
└── cli.py              # CLI: run / check / repl / info / skills / canvas

examples/
├── hello_holo.hl       # Hello, World!
├── laser_config.hl     # Laser + mirror + sensor wiring
├── tensor_graph.hl     # Computation graph pipeline
└── full_system.hl      # Complete system demo

tests/
├── test_lang_lexer.py
├── test_lang_interpreter.py
├── test_tensor.py
├── test_tensor_modules.py  # helpers, ops, matrix, transforms, hyperparams, batch, graph_node
├── test_device.py
├── test_mesh.py
├── test_vm.py
├── test_vm_modules.py      # state, instruction, block
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
