"""Instruction — a single bytecode-level instruction for the HoloLang VM.

Extracted from :mod:`hololang.vm.kernel` so that assemblers, compilers, and
other tooling can build :class:`Instruction` objects without importing the
full :class:`~hololang.vm.kernel.Kernel`.
"""

from __future__ import annotations


class Instruction:
    """A single VM bytecode instruction.

    Parameters
    ----------
    opcode:
        String opcode, e.g. ``"PUSH"``, ``"ADD"``, ``"CALL"``.
    operands:
        Tuple of operand values (ints, floats, strings …).
    """

    __slots__ = ("opcode", "operands")

    def __init__(self, opcode: str, operands: tuple = ()) -> None:
        self.opcode:   str   = opcode
        self.operands: tuple = operands

    def __repr__(self) -> str:
        return f"Instruction({self.opcode!r}, {self.operands!r})"
