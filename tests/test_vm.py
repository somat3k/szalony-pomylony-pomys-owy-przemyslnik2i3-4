"""Tests for the virtual machine subsystem."""

import pytest
from hololang.vm.kernel import Kernel, KernelState, Instruction
from hololang.vm.controller import BlockController
from hololang.vm.runtime import PoolRuntime


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def test_kernel_initial_state():
    k = Kernel("test_k")
    assert k.state == KernelState.IDLE


def test_kernel_simple_program():
    k = Kernel("arith")
    prog = [
        Instruction("PUSH", (3,)),
        Instruction("PUSH", (4,)),
        Instruction("ADD"),
        Instruction("HALT"),
    ]
    k.load(prog)
    result = k.run()
    assert result == 7


def test_kernel_mul():
    k = Kernel("mul")
    prog = [
        Instruction("PUSH", (6,)),
        Instruction("PUSH", (7,)),
        Instruction("MUL"),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 42


def test_kernel_sub():
    k = Kernel("sub")
    prog = [
        Instruction("PUSH", (10,)),
        Instruction("PUSH", (4,)),
        Instruction("SUB"),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 6


def test_kernel_store_load():
    k = Kernel("mem")
    prog = [
        Instruction("PUSH", (99,)),
        Instruction("STORE", ("x",)),
        Instruction("LOAD",  ("x",)),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 99


def test_kernel_jmp():
    k = Kernel("jmp")
    prog = [
        Instruction("JMP",  (2,)),    # → index 2 (PUSH 42)
        Instruction("PUSH", (-1,)),   # should be skipped
        Instruction("PUSH", (42,)),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 42


def test_kernel_jz_not_zero():
    k = Kernel("jz")
    prog = [
        Instruction("PUSH", (1,)),     # truthy → JZ does NOT jump
        Instruction("JZ",   (3,)),
        Instruction("PUSH", (10,)),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 10


def test_kernel_jnz_zero():
    k = Kernel("jnz0")
    prog = [
        Instruction("PUSH", (0,)),     # falsy → JNZ does NOT jump
        Instruction("JNZ",  (3,)),
        Instruction("PUSH", (55,)),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 55


def test_kernel_neg():
    k = Kernel("neg")
    prog = [
        Instruction("PUSH", (5,)),
        Instruction("NEG"),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == -5


def test_kernel_not():
    k = Kernel("not")
    prog = [
        Instruction("PUSH", (False,)),
        Instruction("NOT"),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() is True


def test_kernel_eq():
    k = Kernel("eq")
    prog = [
        Instruction("PUSH", (7,)),
        Instruction("PUSH", (7,)),
        Instruction("EQ"),
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() is True


def test_kernel_dup_swap():
    k = Kernel("dup_swap")
    prog = [
        Instruction("PUSH", (1,)),
        Instruction("PUSH", (2,)),
        Instruction("SWAP"),   # stack: [2, 1]
        Instruction("POP"),    # remove 1 → [2]
        Instruction("HALT"),
    ]
    k.load(prog)
    assert k.run() == 2


def test_kernel_log():
    k = Kernel("logtest")
    prog = [
        Instruction("LOG", ("hello kernel",)),
        Instruction("HALT"),
    ]
    k.load(prog)
    k.run()
    assert any("hello kernel" in m for m in k.get_log())


def test_kernel_stack_overflow():
    k = Kernel("overflow", stack_limit=3)
    prog = [
        Instruction("PUSH", (1,)),
        Instruction("PUSH", (2,)),
        Instruction("PUSH", (3,)),
        Instruction("PUSH", (4,)),   # overflow
    ]
    k.load(prog)
    with pytest.raises(OverflowError):
        k.run()


def test_kernel_replicate():
    k = Kernel("original")
    k.params["power"] = 100
    k2 = k.replicate("clone")
    assert k2.params["power"] == 100
    assert k2.name == "clone"
    assert k2.id != k.id


def test_kernel_unknown_opcode():
    k = Kernel("unk")
    k.load([Instruction("INVALID_OP")])
    with pytest.raises(ValueError):
        k.run()


# ---------------------------------------------------------------------------
# BlockController
# ---------------------------------------------------------------------------

def test_block_controller_simple():
    bc = BlockController("simple")
    bc.add_block("double", fn=lambda x: x * 2)
    result = bc.run(5)
    assert result == 10


def test_block_controller_chain():
    bc = BlockController("chain")
    bc.add_block("add1",  fn=lambda x: x + 1)
    bc.add_block("mul3",  fn=lambda x: x * 3)
    bc.connect("add1", "mul3")
    result = bc.run(4)   # (4+1)*3 = 15
    assert result == 15


def test_block_controller_params():
    bc = BlockController("params")
    bc.add_block("bias", fn=lambda x, b: x + b, b=10)
    result = bc.run(5)
    assert result == 15


def test_block_controller_set_param():
    bc = BlockController("set_param")
    bc.add_block("bias", fn=lambda x, b=0: x + b)
    bc.set_param("bias", b=7)
    result = bc.run(3)
    assert result == 10


def test_block_controller_disable():
    bc = BlockController("disable")
    bc.add_block("double", fn=lambda x: x * 2)
    bc.disable("double")
    result = bc.run(5)
    # Disabled block returns its input unchanged
    assert result == 5


def test_block_controller_generate():
    bc = BlockController("gen")
    bc.add_block("inc", fn=lambda x: x + 1)
    results = list(bc.generate(0, 3))
    assert results == [1, 2, 3]


def test_block_controller_cycle_detection():
    bc = BlockController("cycle")
    bc.add_block("a", fn=lambda x: x)
    bc.add_block("b", fn=lambda x: x)
    bc._edges["a"].append("b")
    bc._edges["b"].append("a")
    with pytest.raises(RuntimeError, match="cycle"):
        bc.run(0)


def test_block_controller_dot():
    bc = BlockController("dot_test")
    bc.add_block("x", fn=lambda v: v)
    dot = bc.to_dot()
    assert "digraph" in dot


# ---------------------------------------------------------------------------
# PoolRuntime
# ---------------------------------------------------------------------------

def test_pool_runtime_create_kernel():
    rt = PoolRuntime("rt")
    k = rt.create_kernel("k1")
    assert k.name == "k1"
    assert "k1" in rt.kernel_names


def test_pool_runtime_replicate():
    rt = PoolRuntime("rt2")
    k = rt.create_kernel("src")
    clones = rt.replicate_kernel("src", 3)
    assert len(clones) == 3
    assert len(rt.kernel_names) == 4  # original + 3


def test_pool_runtime_run_kernel():
    rt = PoolRuntime("rt3")
    k = rt.create_kernel("prog")
    prog = [
        Instruction("PUSH", (21,)),
        Instruction("PUSH", (2,)),
        Instruction("MUL"),
        Instruction("HALT"),
    ]
    k.load(prog)
    result = rt.run_kernel("prog")
    assert result == 42


def test_pool_runtime_run_pipeline():
    rt = PoolRuntime("rt_pipeline")
    bc = BlockController("bc")
    bc.add_block("triple", fn=lambda x: x * 3)
    rt.set_controller(bc)
    results = rt.run_pipeline(initial=4, iterations=2)
    assert results == [12, 36]


def test_pool_runtime_max_kernels():
    rt = PoolRuntime("rt_max", max_kernels=2)
    rt.create_kernel("k1")
    rt.create_kernel("k2")
    with pytest.raises(RuntimeError, match="max_kernels"):
        rt.create_kernel("k3")


def test_pool_runtime_remove_kernel():
    rt = PoolRuntime("rt_rm")
    rt.create_kernel("rem")
    rt.remove_kernel("rem")
    assert "rem" not in rt.kernel_names
