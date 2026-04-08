"""Replicable execution kernel for the HoloLang virtual machine.

Each :class:`Kernel` is an isolated execution context that can be cloned
(:meth:`Kernel.replicate`) to create identical sibling kernels running in
the same :class:`~hololang.vm.runtime.PoolRuntime`.

The kernel's instruction set maps to HoloLang high-level operations.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Kernel state
# ---------------------------------------------------------------------------

class KernelState(Enum):
    IDLE      = "idle"
    RUNNING   = "running"
    SUSPENDED = "suspended"
    FINISHED  = "finished"
    ERROR     = "error"


# ---------------------------------------------------------------------------
# Kernel instruction
# ---------------------------------------------------------------------------

class Instruction:
    """A single bytecode-level instruction."""

    def __init__(self, opcode: str, operands: tuple = ()) -> None:
        self.opcode   = opcode
        self.operands = operands

    def __repr__(self) -> str:
        return f"Instruction({self.opcode}, {self.operands})"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class Kernel:
    """Replicable execution kernel.

    Parameters
    ----------
    name:
        Kernel identifier.
    stack_limit:
        Maximum operand stack depth.
    """

    def __init__(
        self,
        name: str = "",
        stack_limit: int = 256,
    ) -> None:
        self.id:    str = str(uuid.uuid4())[:8]
        self.name:  str = name or f"kernel_{self.id}"
        self.state: KernelState = KernelState.IDLE
        self.params: dict[str, Any] = {}

        # Execution state
        self._stack:        list[Any] = []
        self._stack_limit:  int = stack_limit
        self._registers:    dict[str, Any] = {}
        self._program:      list[Instruction] = []
        self._pc:           int = 0          # programme counter
        self._call_stack:   list[int] = []   # return addresses
        self._lock = threading.Lock()
        self._log:  list[str] = []

        # Registered native operations
        self._ops: dict[str, Callable] = {}
        self._register_default_ops()

    # ------------------------------------------------------------------
    # Default operations
    # ------------------------------------------------------------------

    def _register_default_ops(self) -> None:
        self._ops.update({
            "NOP":   self._op_nop,
            "PUSH":  self._op_push,
            "POP":   self._op_pop,
            "DUP":   self._op_dup,
            "SWAP":  self._op_swap,
            "ADD":   self._op_add,
            "SUB":   self._op_sub,
            "MUL":   self._op_mul,
            "DIV":   self._op_div,
            "MOD":   self._op_mod,
            "NEG":   self._op_neg,
            "AND":   self._op_and,
            "OR":    self._op_or,
            "NOT":   self._op_not,
            "EQ":    self._op_eq,
            "LT":    self._op_lt,
            "GT":    self._op_gt,
            "LOAD":  self._op_load,
            "STORE": self._op_store,
            "JMP":   self._op_jmp,
            "JZ":    self._op_jz,
            "JNZ":   self._op_jnz,
            "CALL":  self._op_call,
            "RET":   self._op_ret,
            "HALT":  self._op_halt,
            "LOG":   self._op_log,
        })

    # ------------------------------------------------------------------
    # Stack helpers
    # ------------------------------------------------------------------

    def _push(self, val: Any) -> None:
        if len(self._stack) >= self._stack_limit:
            raise OverflowError(f"Kernel {self.name}: stack overflow")
        self._stack.append(val)

    def _pop(self) -> Any:
        if not self._stack:
            raise RuntimeError(f"Kernel {self.name}: stack underflow")
        return self._stack.pop()

    def _peek(self) -> Any:
        if not self._stack:
            raise RuntimeError(f"Kernel {self.name}: stack is empty")
        return self._stack[-1]

    # ------------------------------------------------------------------
    # Op implementations
    # ------------------------------------------------------------------

    def _op_nop(self, _instr: Instruction) -> None:
        pass

    def _op_push(self, instr: Instruction) -> None:
        self._push(instr.operands[0])

    def _op_pop(self, _instr: Instruction) -> None:
        self._pop()

    def _op_dup(self, _instr: Instruction) -> None:
        self._push(self._peek())

    def _op_swap(self, _instr: Instruction) -> None:
        a, b = self._pop(), self._pop()
        self._push(a)
        self._push(b)

    def _op_add(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a + b)

    def _op_sub(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a - b)

    def _op_mul(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a * b)

    def _op_div(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a / b)

    def _op_mod(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a % b)

    def _op_neg(self, _instr: Instruction) -> None:
        self._push(-self._pop())

    def _op_and(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a and b)

    def _op_or(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a or b)

    def _op_not(self, _instr: Instruction) -> None:
        self._push(not self._pop())

    def _op_eq(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a == b)

    def _op_lt(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a < b)

    def _op_gt(self, _instr: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(a > b)

    def _op_load(self, instr: Instruction) -> None:
        name = instr.operands[0]
        self._push(self._registers.get(name))

    def _op_store(self, instr: Instruction) -> None:
        name = instr.operands[0]
        self._registers[name] = self._pop()

    def _op_jmp(self, instr: Instruction) -> None:
        self._pc = int(instr.operands[0]) - 1  # -1 because loop will +1

    def _op_jz(self, instr: Instruction) -> None:
        if not self._pop():
            self._pc = int(instr.operands[0]) - 1

    def _op_jnz(self, instr: Instruction) -> None:
        if self._pop():
            self._pc = int(instr.operands[0]) - 1

    def _op_call(self, instr: Instruction) -> None:
        self._call_stack.append(self._pc)
        self._pc = int(instr.operands[0]) - 1

    def _op_ret(self, _instr: Instruction) -> None:
        if self._call_stack:
            self._pc = self._call_stack.pop()
        else:
            self._pc = len(self._program)  # exit

    def _op_halt(self, _instr: Instruction) -> None:
        self.state = KernelState.FINISHED
        self._pc = len(self._program)

    def _op_log(self, instr: Instruction) -> None:
        msg = instr.operands[0] if instr.operands else str(self._peek())
        self._log.append(f"[{self.name}] {msg}")

    # ------------------------------------------------------------------
    # Loading & running programs
    # ------------------------------------------------------------------

    def load(self, program: list[Instruction]) -> None:
        self._program = list(program)
        self._pc = 0
        self.state = KernelState.IDLE
        self._stack.clear()
        self._call_stack.clear()

    def run(self, max_cycles: int = 100_000) -> Any:
        """Execute the loaded program.

        Returns
        -------
        Any
            Top of the operand stack after execution (or ``None``).
        """
        self.state = KernelState.RUNNING
        cycles = 0
        while self._pc < len(self._program):
            if self.state == KernelState.FINISHED:
                break
            if cycles >= max_cycles:
                self.state = KernelState.ERROR
                raise RuntimeError(
                    f"Kernel {self.name}: exceeded max_cycles ({max_cycles})"
                )
            instr = self._program[self._pc]
            handler = self._ops.get(instr.opcode)
            if handler is None:
                raise ValueError(
                    f"Kernel {self.name}: unknown opcode {instr.opcode!r}"
                )
            handler(instr)
            self._pc += 1
            cycles += 1

        self.state = KernelState.FINISHED
        return self._stack[-1] if self._stack else None

    def call_native(self, opcode: str, *args: Any, operands: tuple = ()) -> Any:
        """Directly invoke a native operation by name (for Python callers)."""
        fn = self._ops.get(opcode)
        if fn is None:
            raise ValueError(f"Unknown opcode {opcode!r}")
        for a in args:
            self._push(a)
        fn(Instruction(opcode, operands))
        return self._stack[-1] if self._stack else None

    def register_op(self, opcode: str, fn: Callable) -> None:
        """Register a custom native operation."""
        self._ops[opcode] = fn

    # ------------------------------------------------------------------
    # Replication
    # ------------------------------------------------------------------

    def replicate(self, new_name: str = "") -> "Kernel":
        """Create a deep copy of this kernel with a new identity."""
        clone = Kernel(
            name=new_name or f"{self.name}_clone_{uuid.uuid4().hex[:4]}",
            stack_limit=self._stack_limit,
        )
        clone.params    = copy.deepcopy(self.params)
        clone._program  = list(self._program)
        clone._registers = copy.deepcopy(self._registers)
        # The stack and call-stack are NOT copied – fresh execution state
        return clone

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_log(self) -> list[str]:
        return list(self._log)

    def __repr__(self) -> str:
        return (
            f"Kernel(name={self.name!r}, state={self.state.value}, "
            f"pc={self._pc}, stack={len(self._stack)})"
        )
