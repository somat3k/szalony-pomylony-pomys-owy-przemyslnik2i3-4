"""Tests for the new VM subsystem modules:
vm/state, vm/instruction, vm/block.
"""

from __future__ import annotations

import pytest

from hololang.vm.state import KernelState
from hololang.vm.instruction import Instruction
from hololang.vm.block import Block
from hololang.vm import KernelState as KS_pkg, Instruction as Instr_pkg, Block as Block_pkg


# ---------------------------------------------------------------------------
# KernelState
# ---------------------------------------------------------------------------

class TestKernelState:
    def test_values(self):
        assert KernelState.IDLE.value == "idle"
        assert KernelState.RUNNING.value == "running"
        assert KernelState.SUSPENDED.value == "suspended"
        assert KernelState.FINISHED.value == "finished"
        assert KernelState.ERROR.value == "error"

    def test_re_export_from_vm(self):
        assert KS_pkg.IDLE is KernelState.IDLE

    def test_all_states_reachable(self):
        states = {s.value for s in KernelState}
        assert states == {"idle", "running", "suspended", "finished", "error"}


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

class TestInstruction:
    def test_construction(self):
        instr = Instruction("PUSH", (42,))
        assert instr.opcode == "PUSH"
        assert instr.operands == (42,)

    def test_empty_operands(self):
        instr = Instruction("NOP")
        assert instr.operands == ()

    def test_repr(self):
        instr = Instruction("LOAD", ("x",))
        r = repr(instr)
        assert "LOAD" in r
        assert "x" in r

    def test_re_export_from_vm(self):
        instr = Instr_pkg("ADD")
        assert instr.opcode == "ADD"

    def test_slots(self):
        instr = Instruction("MUL", (3,))
        assert not hasattr(instr, "__dict__")


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class TestBlock:
    def test_basic_execute(self):
        b = Block(name="double", fn=lambda x: x * 2)
        result = b.execute(5)
        assert result == 10
        assert b.last_output == 10

    def test_params_injection(self):
        b = Block(name="add_bias", fn=lambda x, bias=0: x + bias, params={"bias": 7})
        assert b.execute(3) == 10

    def test_disabled_passthrough(self):
        b = Block(name="noop", fn=lambda x: x * 100, enabled=False)
        assert b.execute(42) == 42

    def test_disabled_no_input(self):
        b = Block(name="noop", fn=lambda: "never", enabled=False)
        assert b.execute() is None

    def test_last_output_none_before_execute(self):
        b = Block(name="b", fn=lambda: 1)
        assert b.last_output is None

    def test_re_export_from_vm(self):
        b = Block_pkg(name="test", fn=lambda x: x)
        assert b.execute(99) == 99
