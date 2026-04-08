"""Tests for the HoloLang parser and interpreter."""

import pytest
from hololang.lang.parser import parse, ParseError
from hololang.lang.interpreter import Interpreter, Environment
from hololang.lang.ast_nodes import (
    Program, DeviceDecl, TensorDecl, EnumDecl, FunctionDecl,
    MeshDecl, CanvasDecl, LetStmt, IfStmt, ForStmt, WhileStmt,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_empty():
    prog = parse("")
    assert isinstance(prog, Program)
    assert prog.body == []


def test_parse_let():
    prog = parse("let x = 42")
    assert len(prog.body) == 1
    stmt = prog.body[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.name == "x"


def test_parse_const():
    prog = parse("const PI = 3.14159")
    stmt = prog.body[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.is_const is True


def test_parse_device():
    prog = parse("""
    device Projector {
        type: "laser"
        channels: 3
    }
    """)
    assert len(prog.body) == 1
    assert isinstance(prog.body[0], DeviceDecl)
    assert prog.body[0].name == "Projector"


def test_parse_tensor():
    prog = parse("""
    tensor BeamMatrix[4][4] {
        dtype: "float32"
    }
    """)
    decl = prog.body[0]
    assert isinstance(decl, TensorDecl)
    assert decl.name == "BeamMatrix"
    assert len(decl.dims) == 2


def test_parse_enum():
    prog = parse("""
    enum BeamType {
        RED   = 650,
        GREEN = 532,
        BLUE  = 445
    }
    """)
    decl = prog.body[0]
    assert isinstance(decl, EnumDecl)
    assert decl.name == "BeamType"
    assert len(decl.variants) == 3
    names = [v[0] for v in decl.variants]
    assert "RED" in names
    assert "GREEN" in names


def test_parse_function():
    prog = parse("""
    fn add(a, b) {
        return a + b
    }
    """)
    decl = prog.body[0]
    assert isinstance(decl, FunctionDecl)
    assert decl.name == "add"
    assert len(decl.params) == 2


def test_parse_if_else():
    prog = parse("""
    if x > 0 {
        print("positive")
    } else {
        print("non-positive")
    }
    """)
    assert isinstance(prog.body[0], IfStmt)


def test_parse_for():
    prog = parse("""
    for i in 0..10 {
        print(i)
    }
    """)
    assert isinstance(prog.body[0], ForStmt)


def test_parse_while():
    prog = parse("""
    while x < 100 {
        x = x + 1
    }
    """)
    assert isinstance(prog.body[0], WhileStmt)


def test_parse_annotation():
    prog = parse("""
    @session("demo")
    device Laser {
        power: 50
    }
    """)
    decl = prog.body[0]
    assert isinstance(decl, DeviceDecl)
    assert len(decl.annotations) == 1
    assert decl.annotations[0].name == "session"


def test_parse_mesh():
    prog = parse("""
    mesh HoloMesh(SomeTensor) {
        tile(0, 0) -> 42
    }
    """)
    assert isinstance(prog.body[0], MeshDecl)


def test_parse_pipeline():
    prog = parse("x -> y -> z")
    # Nested pipe expression
    assert len(prog.body) == 1


def test_parse_error_unexpected_token():
    with pytest.raises(ParseError):
        parse("device { no_name_here }")


# ---------------------------------------------------------------------------
# Interpreter tests
# ---------------------------------------------------------------------------

def run(source: str) -> tuple[list[str], Environment]:
    out = []
    env = Environment()
    interp = Interpreter(env=env, output_hook=out.append)
    prog = parse(source)
    interp.eval_program(prog)
    return out, env


def test_print_hello():
    out, _ = run('print("Hello, World!")')
    assert out == ["Hello, World!"]


def test_arithmetic():
    _, env = run("let result = 3 + 4 * 2")
    assert env.lookup("result") == 11


def test_string_concat():
    _, env = run('let s = "hello" + " " + "world"')
    assert env.lookup("s") == "hello world"


def test_if_true():
    out, _ = run("""
    if true {
        print("yes")
    }
    """)
    assert "yes" in out


def test_if_false():
    out, _ = run("""
    if false {
        print("yes")
    } else {
        print("no")
    }
    """)
    assert "no" in out


def test_for_loop():
    _, env = run("""
    let total = 0
    for i in range(5) {
        total = total + i
    }
    """)
    assert env.lookup("total") == 10


def test_while_loop():
    _, env = run("""
    let n = 0
    while n < 5 {
        n = n + 1
    }
    """)
    assert env.lookup("n") == 5


def test_function_call():
    _, env = run("""
    fn square(x) {
        return x * x
    }
    let r = square(7)
    """)
    assert env.lookup("r") == 49


def test_recursive_function():
    _, env = run("""
    fn factorial(n) {
        if n <= 1 {
            return 1
        } else {
            return n * factorial(n - 1)
        }
    }
    let r = factorial(5)
    """)
    assert env.lookup("r") == 120


def test_enum_access():
    _, env = run("""
    enum Color {
        RED   = 0,
        GREEN = 1,
        BLUE  = 2
    }
    let g = Color.GREEN
    """)
    assert env.lookup("g") == 1


def test_device_decl():
    _, env = run("""
    device Projector {
        type: "laser"
        channels: 3
    }
    """)
    from hololang.lang.interpreter import HoloObject
    obj = env.lookup("Projector")
    assert isinstance(obj, HoloObject)
    assert obj.kind == "device"


def test_tensor_decl():
    _, env = run("""
    tensor T[4][4] {
        dtype: "float32"
    }
    """)
    from hololang.tensor.tensor import Tensor
    t = env.lookup("T")
    assert isinstance(t, Tensor)
    assert t.shape == (4, 4)


def test_debug_stmt():
    out, _ = run("debug 42")
    assert any("42" in line for line in out)


def test_list_literal():
    _, env = run("let lst = [1, 2, 3]")
    assert env.lookup("lst") == [1, 2, 3]


def test_dict_literal():
    _, env = run('let d = {"key": 99}')
    assert env.lookup("d") == {"key": 99}


def test_pipeline_expr():
    _, env = run("""
    fn double(x) { return x * 2 }
    let r = 5 -> double
    """)
    assert env.lookup("r") == 10


def test_break_in_while():
    _, env = run("""
    let n = 0
    while true {
        n = n + 1
        if n >= 3 { break }
    }
    """)
    assert env.lookup("n") == 3


def test_continue_in_for():
    _, env = run("""
    let total = 0
    for i in range(6) {
        if i == 3 { continue }
        total = total + i
    }
    """)
    # 0+1+2+4+5 = 12
    assert env.lookup("total") == 12
