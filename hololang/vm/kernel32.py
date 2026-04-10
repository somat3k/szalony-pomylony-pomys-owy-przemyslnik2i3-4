"""32-bit extended runtime virtual machine (Kernel32).

:class:`Kernel32` extends the base :class:`~hololang.vm.kernel.Kernel` with
additional opcodes that align with the CROF blueprint's requirement of
"memory 32-bit extended runtime virtual machine execution" and support
for **Generative Augmented Structures** (GAS) embeddings.

Extended instruction set
------------------------

Arithmetic
~~~~~~~~~~
``INT32_ADD``, ``INT32_SUB``, ``INT32_MUL``, ``INT32_DIV``, ``INT32_MOD``
    Integer arithmetic with 32-bit overflow wrapping (values are masked to
    the signed 32-bit range ``[-2**31, 2**31 - 1]``).

Memory pages
~~~~~~~~~~~~
``PAGE_ALLOC name size``
    Allocate a named memory page of *size* bytes (zero-filled).
``PAGE_FREE name``
    Release a named memory page.
``PAGE_WRITE name offset``
    Pop a bytes value from the stack and write it into a page at *offset*.
``PAGE_READ name offset length``
    Push *length* bytes read from a page at *offset* onto the stack.

Embedding (GAS)
~~~~~~~~~~~~~~~
``EMB_LOAD space_name key``
    Load an embedding vector from the named :class:`~hololang.tensor.embeddings.EmbeddingSpace`
    and push it as a :class:`~hololang.tensor.tensor.Tensor` onto the stack.
``EMB_STORE space_name key``
    Pop a :class:`~hololang.tensor.tensor.Tensor` from the stack and store it
    in the named :class:`~hololang.tensor.embeddings.EmbeddingSpace`.
``EMB_SIM space_name key``
    Pop a query :class:`~hololang.tensor.tensor.Tensor` from the stack, find
    the cosine-similarity to *key* in *space_name*, and push the score.
``EMB_KNN space_name k``
    Pop a query :class:`~hololang.tensor.tensor.Tensor` from the stack, run
    k-nearest-neighbours in *space_name*, and push the list of ``(key, score)``
    pairs.
``EMB_COMPOSE_MEAN space_name``
    Pop a list of key strings from the stack, compute the mean embedding, and
    push the result.

All opcodes from the base :class:`~hololang.vm.kernel.Kernel` are inherited.
"""

from __future__ import annotations

import struct
from typing import Any

from hololang.vm.kernel import Kernel, Instruction, KernelState
from hololang.tensor.embeddings import EmbeddingSpace
from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# 32-bit overflow helper
# ---------------------------------------------------------------------------

_INT32_MIN = -(2 ** 31)
_INT32_MAX =   2 ** 31 - 1


def _wrap32(n: int) -> int:
    """Wrap *n* to the signed 32-bit integer range."""
    n = int(n) & 0xFFFFFFFF
    if n >= 0x80000000:
        n -= 0x100000000
    return n


# ---------------------------------------------------------------------------
# Kernel32
# ---------------------------------------------------------------------------

class Kernel32(Kernel):
    """32-bit extended runtime kernel with embedding (GAS) ops.

    Parameters
    ----------
    name:
        Kernel identifier.
    stack_limit:
        Maximum operand stack depth.
    embedding_spaces:
        Pre-populated :class:`~hololang.tensor.embeddings.EmbeddingSpace`
        objects available to ``EMB_*`` opcodes.  More spaces can be
        registered at runtime via :meth:`register_embedding_space`.
    """

    def __init__(
        self,
        name: str = "",
        stack_limit: int = 256,
        embedding_spaces: dict[str, EmbeddingSpace] | None = None,
    ) -> None:
        super().__init__(name=name, stack_limit=stack_limit)
        # Named memory pages: name -> bytearray
        self._pages: dict[str, bytearray] = {}
        # GAS embedding spaces: name -> EmbeddingSpace
        self._emb_spaces: dict[str, EmbeddingSpace] = dict(embedding_spaces or {})
        self._register_extended_ops()

    # ------------------------------------------------------------------
    # Embedding space management
    # ------------------------------------------------------------------

    def register_embedding_space(self, space: EmbeddingSpace) -> None:
        """Register an :class:`~hololang.tensor.embeddings.EmbeddingSpace`."""
        self._emb_spaces[space.name] = space

    def get_embedding_space(self, name: str) -> EmbeddingSpace | None:
        return self._emb_spaces.get(name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_extended_ops(self) -> None:
        self._ops.update({
            # 32-bit arithmetic
            "INT32_ADD":  self._op_int32_add,
            "INT32_SUB":  self._op_int32_sub,
            "INT32_MUL":  self._op_int32_mul,
            "INT32_DIV":  self._op_int32_div,
            "INT32_MOD":  self._op_int32_mod,
            # Memory pages
            "PAGE_ALLOC": self._op_page_alloc,
            "PAGE_FREE":  self._op_page_free,
            "PAGE_WRITE": self._op_page_write,
            "PAGE_READ":  self._op_page_read,
            # GAS / embedding
            "EMB_LOAD":          self._op_emb_load,
            "EMB_STORE":         self._op_emb_store,
            "EMB_SIM":           self._op_emb_sim,
            "EMB_KNN":           self._op_emb_knn,
            "EMB_COMPOSE_MEAN":  self._op_emb_compose_mean,
        })

    # ------------------------------------------------------------------
    # 32-bit arithmetic ops
    # ------------------------------------------------------------------

    def _op_int32_add(self, _: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(_wrap32(int(a) + int(b)))

    def _op_int32_sub(self, _: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(_wrap32(int(a) - int(b)))

    def _op_int32_mul(self, _: Instruction) -> None:
        b, a = self._pop(), self._pop()
        self._push(_wrap32(int(a) * int(b)))

    def _op_int32_div(self, _: Instruction) -> None:
        b, a = self._pop(), self._pop()
        if int(b) == 0:
            raise ZeroDivisionError("INT32_DIV: division by zero")
        self._push(_wrap32(int(a) // int(b)))

    def _op_int32_mod(self, _: Instruction) -> None:
        b, a = self._pop(), self._pop()
        if int(b) == 0:
            raise ZeroDivisionError("INT32_MOD: modulo by zero")
        self._push(_wrap32(int(a) % int(b)))

    # ------------------------------------------------------------------
    # Memory page ops
    # ------------------------------------------------------------------

    def _op_page_alloc(self, instr: Instruction) -> None:
        name = str(instr.operands[0])
        size = int(instr.operands[1])
        self._pages[name] = bytearray(size)

    def _op_page_free(self, instr: Instruction) -> None:
        name = str(instr.operands[0])
        self._pages.pop(name, None)

    def _op_page_write(self, instr: Instruction) -> None:
        name   = str(instr.operands[0])
        offset = int(instr.operands[1])
        data   = self._pop()
        if not isinstance(data, (bytes, bytearray)):
            if isinstance(data, (int, float)):
                data = struct.pack("<f", float(data))
            else:
                data = str(data).encode()
        if name not in self._pages:
            raise RuntimeError(f"PAGE_WRITE: unknown page {name!r}")
        page = self._pages[name]
        end  = offset + len(data)
        if end > len(page):
            raise OverflowError(
                f"PAGE_WRITE: write [{offset}:{end}] exceeds page size {len(page)}"
            )
        page[offset:end] = data

    def _op_page_read(self, instr: Instruction) -> None:
        name   = str(instr.operands[0])
        offset = int(instr.operands[1])
        length = int(instr.operands[2])
        if name not in self._pages:
            raise RuntimeError(f"PAGE_READ: unknown page {name!r}")
        page = self._pages[name]
        self._push(bytes(page[offset:offset + length]))

    # ------------------------------------------------------------------
    # GAS / embedding ops
    # ------------------------------------------------------------------

    def _resolve_space(self, name: str) -> EmbeddingSpace:
        space = self._emb_spaces.get(name)
        if space is None:
            raise RuntimeError(f"EMB: no embedding space named {name!r}")
        return space

    def _op_emb_load(self, instr: Instruction) -> None:
        space_name = str(instr.operands[0])
        key        = str(instr.operands[1])
        space      = self._resolve_space(space_name)
        vec        = space.require(key)
        self._push(vec)

    def _op_emb_store(self, instr: Instruction) -> None:
        space_name = str(instr.operands[0])
        key        = str(instr.operands[1])
        vec        = self._pop()
        space      = self._resolve_space(space_name)
        space.set(key, vec)

    def _op_emb_sim(self, instr: Instruction) -> None:
        space_name = str(instr.operands[0])
        key        = str(instr.operands[1])
        query: Tensor = self._pop()
        space  = self._resolve_space(space_name)
        target = space.require(key)
        score  = space.cosine_similarity(query, target)
        self._push(score)

    def _op_emb_knn(self, instr: Instruction) -> None:
        space_name = str(instr.operands[0])
        k          = int(instr.operands[1])
        query: Tensor = self._pop()
        space  = self._resolve_space(space_name)
        result = space.knn(query, k=k)
        self._push(result)

    def _op_emb_compose_mean(self, instr: Instruction) -> None:
        space_name = str(instr.operands[0])
        keys: list[str] = self._pop()
        space  = self._resolve_space(space_name)
        result = space.compose_mean(keys)
        self._push(result)

    # ------------------------------------------------------------------
    # Replicate (override to preserve Kernel32-specific state)
    # ------------------------------------------------------------------

    def replicate(self, new_name: str = "") -> "Kernel32":
        import copy
        clone = Kernel32(
            name             = new_name or f"{self.name}_clone",
            stack_limit      = self._stack_limit,
            embedding_spaces = {k: v for k, v in self._emb_spaces.items()},
        )
        clone.params     = copy.deepcopy(self.params)
        clone._program   = list(self._program)
        clone._registers = copy.deepcopy(self._registers)
        clone._pages     = {k: bytearray(v) for k, v in self._pages.items()}
        return clone

    def __repr__(self) -> str:
        return (
            f"Kernel32(name={self.name!r}, state={self.state.value}, "
            f"pc={self._pc}, pages={len(self._pages)}, "
            f"emb_spaces={len(self._emb_spaces)})"
        )
