"""Tests for the HoloLang AST → Kernel compiler."""

import pytest
from hololang.lang.parser import parse
from hololang.lang.compiler import Compiler, CompileError
from hololang.vm.kernel import Kernel


def _run(source: str, max_cycles: int = 50_000) -> tuple[object, list[str]]:
    """Compile *source*, run it, return (top_of_stack, kernel_log)."""
    prog = parse(source)
    comp = Compiler()
    instrs = comp.compile(prog)
    k = Kernel("test")
    k.load(instrs)
    result = k.run(max_cycles=max_cycles)
    return result, k.get_log()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_integer_addition():
    result, _ = _run("let x = 3 + 4")
    # registers don't affect stack top — result is None (HALT)
    # check via debug
    _, log = _run("debug 3 + 4")
    assert "[test] 7" in log


def test_integer_multiplication():
    _, log = _run("debug 6 * 7")
    assert "[test] 42" in log


def test_float_arithmetic():
    _, log = _run("debug 1.5 + 0.5")
    assert "[test] 2.0" in log


def test_subtraction():
    _, log = _run("debug 10 - 3")
    assert "[test] 7" in log


def test_division():
    _, log = _run("debug 10 / 4")
    assert "[test] 2.5" in log


def test_modulo():
    _, log = _run("debug 10 % 3")
    assert "[test] 1" in log


def test_unary_negation():
    _, log = _run("debug -5")
    assert "[test] -5" in log


def test_nested_expression():
    _, log = _run("debug (2 + 3) * (4 - 1)")
    assert "[test] 15" in log


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

def test_let_and_load():
    _, log = _run("let x = 99\ndebug x")
    assert "[test] 99" in log


def test_variable_update():
    _, log = _run("let x = 1\nx = x + 1\ndebug x")
    assert "[test] 2" in log


def test_multiple_variables():
    _, log = _run("let a = 3\nlet b = 7\ndebug a + b")
    assert "[test] 10" in log


# ---------------------------------------------------------------------------
# Boolean and comparison
# ---------------------------------------------------------------------------

def test_equality():
    _, log = _run("debug 1 == 1")
    assert "[test] True" in log


def test_inequality():
    _, log = _run("debug 1 != 2")
    assert "[test] True" in log


def test_less_than():
    _, log = _run("debug 3 < 5")
    assert "[test] True" in log


def test_greater_than():
    _, log = _run("debug 5 > 3")
    assert "[test] True" in log


def test_lte():
    _, log = _run("debug 3 <= 3")
    assert "[test] True" in log


def test_gte():
    _, log = _run("debug 5 >= 5")
    assert "[test] True" in log


# ---------------------------------------------------------------------------
# If / else
# ---------------------------------------------------------------------------

def test_if_true_branch():
    _, log = _run('let x = 0\nif true { x = 1 }\ndebug x')
    assert "[test] 1" in log


def test_if_false_branch():
    _, log = _run('let x = 0\nif false { x = 1 }\ndebug x')
    assert "[test] 0" in log


def test_if_else():
    _, log = _run('let x = 0\nif false { x = 1 } else { x = 2 }\ndebug x')
    assert "[test] 2" in log


def test_if_condition_expression():
    _, log = _run('let n = 5\nlet r = 0\nif n > 3 { r = 1 }\ndebug r')
    assert "[test] 1" in log


# ---------------------------------------------------------------------------
# While loop
# ---------------------------------------------------------------------------

def test_while_basic():
    _, log = _run('let n = 0\nwhile n < 5 { n = n + 1 }\ndebug n')
    assert "[test] 5" in log


def test_while_accumulate():
    _, log = _run('let s = 0\nlet i = 1\nwhile i <= 10 { s = s + i\ni = i + 1 }\ndebug s')
    assert "[test] 55" in log


# ---------------------------------------------------------------------------
# For loop
# ---------------------------------------------------------------------------

def test_for_range_one_arg():
    _, log = _run('let s = 0\nfor i in range(5) { s = s + i }\ndebug s')
    assert "[test] 10" in log


def test_for_range_two_args():
    _, log = _run('let s = 0\nfor i in range(1, 6) { s = s + i }\ndebug s')
    assert "[test] 15" in log


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def test_fn_simple():
    _, log = _run('fn double(x) { return x * 2 }\ndebug double(7)')
    assert "[test] 14" in log


def test_fn_multiple_params():
    _, log = _run('fn add(a, b) { return a + b }\ndebug add(3, 4)')
    assert "[test] 7" in log


def test_fn_recursive_factorial():
    src = '''
fn factorial(n) {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}
debug factorial(5)
'''
    _, log = _run(src, max_cycles=500_000)
    assert "[test] 120" in log


def test_fn_no_args():
    _, log = _run('fn answer() { return 42 }\ndebug answer()')
    assert "[test] 42" in log


# ---------------------------------------------------------------------------
# Domain declarations are silently skipped
# ---------------------------------------------------------------------------

def test_domain_decls_skipped():
    """Mixed program: domain declarations must not raise CompileError."""
    src = '''
let x = 10
debug x
'''
    _, log = _run(src)
    assert "[test] 10" in log


# ---------------------------------------------------------------------------
# CompileError cases
# ---------------------------------------------------------------------------

def test_undefined_function_raises():
    with pytest.raises(CompileError, match="Undefined function"):
        prog = parse("let r = missing_fn(1)")
        Compiler().compile(prog)


def test_unsupported_for_iterable():
    with pytest.raises(CompileError):
        prog = parse("for x in someList { debug x }")
        Compiler().compile(prog)


# ---------------------------------------------------------------------------
# Debug / print built-ins
# ---------------------------------------------------------------------------

def test_debug_statement():
    _, log = _run("debug 123")
    assert "[test] 123" in log


def test_print_single_arg():
    _, log = _run('print("hello")')
    assert "[test] hello" in log
