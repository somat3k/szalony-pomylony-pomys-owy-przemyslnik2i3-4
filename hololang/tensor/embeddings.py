"""GAS — Generative Augmented Structures: embedding memory for HoloLang.

An :class:`EmbeddingSpace` is a lookup table of fixed-dimension float32
vectors that can be queried, composed, and stored in a 32-bit memory page
usable by the extended VM (:class:`~hololang.vm.kernel32.Kernel32`).

Each embedding is a :class:`~hololang.tensor.tensor.Tensor` of shape
``(dim,)`` stored under a string key.  The module supports:

* Key-value lookup (``get`` / ``set``)
* Cosine and dot-product similarity search (``most_similar``)
* Vector composition (``compose_mean``, ``compose_sum``)
* Nearest-neighbour batch search (``knn``)
* Persistence via :meth:`EmbeddingSpace.to_dict` / :meth:`EmbeddingSpace.from_dict`
* 32-bit memory page serialisation (``to_memory_page`` / ``from_memory_page``)

The module follows the blueprint requirement of "embeddings in forms of
memory 32-bit extended runtime virtual machine execution": each embedding
dimension is stored as an IEEE-754 float32, and the memory page is a flat
``bytes`` object with a 32-byte header followed by raw float32 data.
"""

from __future__ import annotations

import math
import struct
import json
from dataclasses import dataclass, field
from typing import Any

from hololang.tensor.tensor import Tensor, _NP   # noqa: PLC2701  (use module private)


# ---------------------------------------------------------------------------
# Memory page layout (32-bit, IEEE-754 float32)
#
#   Offset  Size  Field
#   0       4     Magic = 0x474153_20 ("GAS ")
#   4       4     Version = 1 (uint32)
#   8       4     dim     (uint32)
#   12      4     count   (uint32)
#   16      4     key_stride (bytes per key, padded to 4) (uint32)
#   20      12    reserved (zeros)
#   32      …     key block  (count × key_stride bytes, UTF-8 NUL-padded)
#   32+key… …     data block (count × dim × 4 bytes, float32 little-endian)
# ---------------------------------------------------------------------------

_MAGIC   = b"GAS "
_VERSION = 1
_HEADER  = 32
_KEY_STRIDE = 64  # bytes reserved per key (max 63 chars + NUL)


@dataclass
class EmbeddingEntry:
    """A single embedding in the space."""
    key:    str
    vector: Tensor   # shape (dim,)

    def __repr__(self) -> str:
        return f"EmbeddingEntry(key={self.key!r}, dim={self.vector.size})"


class EmbeddingSpace:
    """Fixed-dimension float32 embedding table.

    Parameters
    ----------
    dim:
        Embedding dimension.  All stored vectors must have exactly *dim*
        elements.
    name:
        Human-readable label for the space.
    """

    def __init__(self, dim: int, name: str = "emb") -> None:
        if dim < 1:
            raise ValueError("EmbeddingSpace: dim must be ≥ 1")
        self.dim:  int  = dim
        self.name: str  = name
        self._table: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, vector: "Tensor | list[float]") -> None:
        """Store a vector under *key*.  Raises ``ValueError`` on dim mismatch."""
        if isinstance(vector, list):
            vector = Tensor([self.dim], vector)
        if vector.size != self.dim:
            raise ValueError(
                f"EmbeddingSpace({self.name}): expected dim={self.dim}, "
                f"got {vector.size}"
            )
        self._table[key] = vector

    def get(self, key: str) -> Tensor | None:
        return self._table.get(key)

    def require(self, key: str) -> Tensor:
        v = self._table.get(key)
        if v is None:
            raise KeyError(f"EmbeddingSpace: key {key!r} not found")
        return v

    def remove(self, key: str) -> None:
        self._table.pop(key, None)

    @property
    def keys(self) -> list[str]:
        return list(self._table.keys())

    @property
    def count(self) -> int:
        return len(self._table)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    @staticmethod
    def _dot(a: Tensor, b: Tensor) -> float:
        return sum(x * y for x, y in zip(a._data, b._data))

    @staticmethod
    def _norm(v: Tensor) -> float:
        return math.sqrt(sum(x * x for x in v._data))

    def cosine_similarity(self, a: Tensor, b: Tensor) -> float:
        """Return the cosine similarity in [-1, 1]."""
        na, nb = self._norm(a), self._norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return self._dot(a, b) / (na * nb)

    def most_similar(
        self,
        query: "Tensor | list[float]",
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Return the *top_k* most similar keys to *query*.

        Parameters
        ----------
        query:
            Query vector (same dim as the space).
        top_k:
            Number of results to return.
        metric:
            ``"cosine"`` (default) or ``"dot"``.

        Returns
        -------
        list of ``(key, score)`` sorted descending by score.
        """
        if isinstance(query, list):
            query = Tensor([self.dim], query)

        scores: list[tuple[str, float]] = []
        for key, vec in self._table.items():
            if metric == "dot":
                score = self._dot(query, vec)
            else:
                score = self.cosine_similarity(query, vec)
            scores.append((key, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def knn(
        self,
        query: "Tensor | list[float]",
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Alias for :meth:`most_similar` with ``metric="cosine"``."""
        return self.most_similar(query, top_k=k, metric="cosine")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose_mean(self, keys: list[str]) -> Tensor:
        """Return the element-wise mean of the named embeddings."""
        vecs = [self.require(k) for k in keys]
        n = len(vecs)
        if n == 0:
            return Tensor([self.dim])
        result = [0.0] * self.dim
        for v in vecs:
            for i, x in enumerate(v._data):
                result[i] += x
        return Tensor([self.dim], [x / n for x in result])

    def compose_sum(self, keys: list[str]) -> Tensor:
        """Return the element-wise sum of the named embeddings."""
        vecs = [self.require(k) for k in keys]
        if not vecs:
            return Tensor([self.dim])
        result = [0.0] * self.dim
        for v in vecs:
            for i, x in enumerate(v._data):
                result[i] += x
        return Tensor([self.dim], result)

    # ------------------------------------------------------------------
    # Serialisation (dict / JSON)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":  self.name,
            "dim":   self.dim,
            "table": {k: list(v._data) for k, v in self._table.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingSpace":
        space = cls(dim=d["dim"], name=d.get("name", "emb"))
        for k, data in d.get("table", {}).items():
            space.set(k, data)
        return space

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "EmbeddingSpace":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # 32-bit memory page I/O
    #
    # Page format (little-endian):
    #   Header (32 bytes):  magic(4) version(4) dim(4) count(4) key_stride(4) pad(12)
    #   Key block:          count × _KEY_STRIDE bytes (UTF-8, NUL-padded)
    #   Data block:         count × dim × 4 bytes    (float32 LE)
    # ------------------------------------------------------------------

    def to_memory_page(self) -> bytes:
        """Serialise the embedding table to a 32-bit memory page (bytes)."""
        keys   = sorted(self._table.keys())
        count  = len(keys)
        # Header
        header = struct.pack(
            "<4sIIII12s",
            _MAGIC,
            _VERSION,
            self.dim,
            count,
            _KEY_STRIDE,
            b"\x00" * 12,
        )
        # Key block
        key_block = b""
        for k in keys:
            encoded = k.encode("utf-8")[:_KEY_STRIDE - 1]
            key_block += encoded.ljust(_KEY_STRIDE, b"\x00")
        # Data block (float32)
        data_block = b""
        for k in keys:
            vec = self._table[k]
            data_block += struct.pack(f"<{self.dim}f", *vec._data)
        return header + key_block + data_block

    @classmethod
    def from_memory_page(cls, page: bytes, name: str = "emb") -> "EmbeddingSpace":
        """Deserialise an embedding table from a 32-bit memory page."""
        if len(page) < _HEADER:
            raise ValueError("Memory page too short for GAS header")
        magic, version, dim, count, key_stride, _ = struct.unpack_from(
            "<4sIIII12s", page, 0
        )
        if magic != _MAGIC:
            raise ValueError(f"Bad GAS magic: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported GAS page version {version}")

        space = cls(dim=dim, name=name)
        key_offset  = _HEADER
        data_offset = _HEADER + count * key_stride

        for i in range(count):
            raw_key = page[key_offset:key_offset + key_stride]
            key     = raw_key.rstrip(b"\x00").decode("utf-8", errors="replace")
            key_offset += key_stride

            raw_floats = page[data_offset:data_offset + dim * 4]
            data_offset += dim * 4
            floats = list(struct.unpack(f"<{dim}f", raw_floats))
            space.set(key, floats)

        return space

    def __repr__(self) -> str:
        return f"EmbeddingSpace(name={self.name!r}, dim={self.dim}, count={self.count})"
