"""Integration test: HoloRuntime end-to-end."""

import pytest
from hololang.runtime import HoloRuntime


def run(source: str) -> tuple[list[str], HoloRuntime]:
    """Helper: run HoloLang source and return (output_lines, runtime)."""
    rt = HoloRuntime(session_name="test_session")
    rt.run_string(source)
    return rt.get_output(), rt


def test_runtime_hello():
    out, _ = run('print("Hello, HoloLang!")')
    assert "Hello, HoloLang!" in out


def test_runtime_device():
    _, rt = run("""
    device Laser {
        type: "green-laser"
        power: 50
    }
    """)
    from hololang.lang.interpreter import HoloObject
    laser = rt.get("Laser")
    assert laser is not None
    assert isinstance(laser, HoloObject)


def test_runtime_tensor():
    _, rt = run("""
    tensor T[3][3] {
        dtype: "float32"
    }
    """)
    from hololang.tensor.tensor import Tensor
    t = rt.get("T")
    assert isinstance(t, Tensor)
    assert t.shape == (3, 3)


def test_runtime_enum():
    _, rt = run("""
    enum Mode {
        FAST = 0,
        SLOW = 1
    }
    let m = Mode.FAST
    """)
    assert rt.get("m") == 0


def test_runtime_mesh():
    _, rt = run("""
    mesh MyCanvas {
        tile(0, 0) -> 10
        tile(0, 1) -> 20
    }
    """)
    from hololang.mesh.canvas import Canvas
    canvas = rt.get("MyCanvas")
    assert isinstance(canvas, Canvas)
    assert len(canvas) == 2


def test_runtime_session_logging():
    _, rt = run('print("test")')
    events = rt.session.get_events()
    assert any(e.event_type == "run_string" for e in events)


def test_runtime_channel():
    _, rt = run("""
    channel C(grpc) {
        host: "localhost"
        port: 50051
    }
    """)
    from hololang.network.api import Channel
    ch = rt.get("C")
    assert isinstance(ch, Channel)


def test_runtime_run_file(tmp_path):
    hl_file = tmp_path / "test.hl"
    hl_file.write_text('print("from file")', encoding="utf-8")
    rt = HoloRuntime()
    rt.run_file(str(hl_file))
    assert "from file" in rt.get_output()


def test_runtime_run_file_not_found():
    rt = HoloRuntime()
    with pytest.raises(FileNotFoundError):
        rt.run_file("/nonexistent/path/to/program.hl")


def test_runtime_builtin_math():
    _, rt = run("""
    let r = sqrt(16.0)
    let s = abs(-5)
    """)
    assert abs(rt.get("r") - 4.0) < 1e-9
    assert rt.get("s") == 5


def test_runtime_context_manager():
    with HoloRuntime() as rt:
        rt.run_string('print("ctx")')
    # Session should be ended
    assert rt.session.ended_at is not None


def test_runtime_display_canvas():
    _, rt = run("""
    mesh VC {
        tile(0, 0) -> "A"
        tile(0, 1) -> "B"
    }
    """)
    canvas_art = rt.display_canvas(use_color=False)
    assert isinstance(canvas_art, str)


def test_runtime_function_and_loop():
    out, _ = run("""
    fn greet(name) {
        print("Hello " + name)
    }
    let names = ["Alice", "Bob", "Carol"]
    for n in names {
        greet(n)
    }
    """)
    assert "Hello Alice" in out
    assert "Hello Bob" in out
    assert "Hello Carol" in out


def test_runtime_example_hello(tmp_path):
    """Smoke-test: run the hello_holo.hl example program."""
    import pathlib
    example = pathlib.Path(__file__).parent.parent / "examples" / "hello_holo.hl"
    rt = HoloRuntime(session_name="example_test")
    rt.run_file(str(example))
    out = rt.get_output()
    assert any("Hello" in line for line in out)
