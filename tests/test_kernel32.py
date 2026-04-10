"""Tests for Kernel32 extended VM (vm/kernel32.py)."""

import pytest
from hololang.vm.kernel import Instruction
from hololang.vm.kernel32 import Kernel32, _wrap32
from hololang.tensor.embeddings import EmbeddingSpace
from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# 32-bit overflow wrapping
# ---------------------------------------------------------------------------

def test_wrap32_positive():
    assert _wrap32(100) == 100


def test_wrap32_max():
    assert _wrap32(2**31 - 1) == 2**31 - 1


def test_wrap32_overflow():
    assert _wrap32(2**31) == -(2**31)


def test_wrap32_negative():
    assert _wrap32(-1) == -1


# ---------------------------------------------------------------------------
# INT32 arithmetic opcodes
# ---------------------------------------------------------------------------

def _k32(*instrs):
    k = Kernel32("t32")
    k.load(list(instrs) + [Instruction("HALT")])
    return k.run(), k


def test_int32_add():
    result, _ = _k32(
        Instruction("PUSH", (2**31 - 1,)),
        Instruction("PUSH", (1,)),
        Instruction("INT32_ADD"),
    )
    assert result == -(2**31)   # overflow wrap


def test_int32_sub():
    result, _ = _k32(
        Instruction("PUSH", (5,)),
        Instruction("PUSH", (3,)),
        Instruction("INT32_SUB"),
    )
    assert result == 2


def test_int32_mul():
    result, _ = _k32(
        Instruction("PUSH", (6,)),
        Instruction("PUSH", (7,)),
        Instruction("INT32_MUL"),
    )
    assert result == 42


def test_int32_div():
    result, _ = _k32(
        Instruction("PUSH", (10,)),
        Instruction("PUSH", (3,)),
        Instruction("INT32_DIV"),
    )
    assert result == 3


def test_int32_mod():
    result, _ = _k32(
        Instruction("PUSH", (10,)),
        Instruction("PUSH", (3,)),
        Instruction("INT32_MOD"),
    )
    assert result == 1


def test_int32_div_zero():
    k = Kernel32("div_zero")
    k.load([
        Instruction("PUSH", (1,)),
        Instruction("PUSH", (0,)),
        Instruction("INT32_DIV"),
        Instruction("HALT"),
    ])
    with pytest.raises(ZeroDivisionError):
        k.run()


# ---------------------------------------------------------------------------
# Memory pages
# ---------------------------------------------------------------------------

def test_page_alloc_write_read():
    k = Kernel32("pages")
    k.load([
        Instruction("PAGE_ALLOC", ("pg", 64)),
        Instruction("PUSH", (b"\xAB\xCD",)),
        Instruction("PAGE_WRITE", ("pg", 0)),
        Instruction("PAGE_READ", ("pg", 0, 2)),
        Instruction("HALT"),
    ])
    result = k.run()
    assert result == b"\xAB\xCD"


def test_page_write_overflow_raises():
    k = Kernel32("pg_overflow")
    k.load([
        Instruction("PAGE_ALLOC", ("pg", 4)),
        Instruction("PUSH", (b"\x01\x02\x03\x04\x05",)),  # 5 bytes into 4
        Instruction("PAGE_WRITE", ("pg", 0)),
        Instruction("HALT"),
    ])
    with pytest.raises(OverflowError):
        k.run()


def test_page_free():
    k = Kernel32("pgfree")
    k.load([
        Instruction("PAGE_ALLOC", ("pg", 8)),
        Instruction("PAGE_FREE", ("pg",)),
        Instruction("HALT"),
    ])
    k.run()
    assert "pg" not in k._pages


def test_page_unknown_raises():
    k = Kernel32("pg_unknown")
    k.load([
        Instruction("PAGE_READ", ("missing", 0, 1)),
        Instruction("HALT"),
    ])
    with pytest.raises(RuntimeError, match="unknown page"):
        k.run()


# ---------------------------------------------------------------------------
# GAS / Embedding opcodes
# ---------------------------------------------------------------------------

def _setup_space(dim=3):
    space = EmbeddingSpace(dim=dim, name="emb")
    space.set("cat", [1.0, 0.0, 0.0])
    space.set("dog", [0.0, 1.0, 0.0])
    space.set("fish", [0.0, 0.0, 1.0])
    return space


def test_emb_load():
    space = _setup_space()
    k = Kernel32("emb_load", embedding_spaces={"emb": space})
    k.load([
        Instruction("EMB_LOAD", ("emb", "cat")),
        Instruction("HALT"),
    ])
    result = k.run()
    assert isinstance(result, Tensor)
    assert result._data == [1.0, 0.0, 0.0]


def test_emb_store():
    space = EmbeddingSpace(dim=3, name="emb")
    k = Kernel32("emb_store", embedding_spaces={"emb": space})
    k.load([
        Instruction("PUSH", (Tensor([3], [9.0, 8.0, 7.0]),)),
        Instruction("EMB_STORE", ("emb", "newkey")),
        Instruction("HALT"),
    ])
    k.run()
    vec = space.get("newkey")
    assert vec is not None
    assert vec._data == [9.0, 8.0, 7.0]


def test_emb_sim():
    space = _setup_space()
    k = Kernel32("emb_sim", embedding_spaces={"emb": space})
    k.load([
        Instruction("PUSH", (Tensor([3], [1.0, 0.0, 0.0]),)),
        Instruction("EMB_SIM", ("emb", "cat")),
        Instruction("HALT"),
    ])
    score = k.run()
    assert abs(score - 1.0) < 1e-6


def test_emb_knn():
    space = _setup_space()
    k = Kernel32("emb_knn", embedding_spaces={"emb": space})
    k.load([
        Instruction("PUSH", (Tensor([3], [1.0, 0.0, 0.0]),)),
        Instruction("EMB_KNN", ("emb", 2)),
        Instruction("HALT"),
    ])
    result = k.run()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0][0] == "cat"


def test_emb_compose_mean():
    space = _setup_space()
    k = Kernel32("emb_mean", embedding_spaces={"emb": space})
    k.load([
        Instruction("PUSH", (["cat", "dog"],)),
        Instruction("EMB_COMPOSE_MEAN", ("emb",)),
        Instruction("HALT"),
    ])
    result = k.run()
    assert isinstance(result, Tensor)
    assert abs(result._data[0] - 0.5) < 1e-6
    assert abs(result._data[1] - 0.5) < 1e-6


def test_emb_unknown_space_raises():
    k = Kernel32("emb_missing")
    k.load([
        Instruction("PUSH", (Tensor([2], [1.0, 0.0]),)),
        Instruction("EMB_SIM", ("no_space", "key")),
        Instruction("HALT"),
    ])
    with pytest.raises(RuntimeError, match="no embedding space"):
        k.run()


# ---------------------------------------------------------------------------
# register_embedding_space + replicate
# ---------------------------------------------------------------------------

def test_register_embedding_space():
    k = Kernel32("reg_emb")
    space = EmbeddingSpace(dim=2, name="new_emb")
    space.set("x", [1.0, 0.0])
    k.register_embedding_space(space)
    assert k.get_embedding_space("new_emb") is space


def test_replicate_preserves_spaces():
    space = EmbeddingSpace(dim=2, name="my_emb")
    space.set("x", [1.0, 0.0])
    k = Kernel32("orig", embedding_spaces={"my_emb": space})
    clone = k.replicate("clone")
    assert clone.get_embedding_space("my_emb") is not None


def test_inherited_opcodes():
    """Kernel32 must still support base Kernel opcodes."""
    k = Kernel32("base_ops")
    k.load([
        Instruction("PUSH", (10,)),
        Instruction("PUSH", (5,)),
        Instruction("ADD"),
        Instruction("HALT"),
    ])
    assert k.run() == 15
