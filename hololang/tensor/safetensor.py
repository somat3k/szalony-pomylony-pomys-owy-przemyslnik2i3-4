"""SafeTensor – a validated, metadata-rich wrapper around :class:`Tensor`.

SafeTensor enforces:
* shape constraints (min/max per dimension)
* dtype whitelist
* value range clamping or rejection
* serialization / deserialization with integrity checks (CRC32)

This enables safe use of tensors in holographic pipelines where
out-of-range values could damage physical devices.
"""

from __future__ import annotations

import json
import zlib
from typing import Any

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# Dtype registry
# ---------------------------------------------------------------------------

_ALLOWED_DTYPES = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
})


# ---------------------------------------------------------------------------
# SafeTensor
# ---------------------------------------------------------------------------

class SafeTensor:
    """Tensor with safety constraints and serialization support.

    Parameters
    ----------
    tensor:
        Underlying :class:`~hololang.tensor.tensor.Tensor`.
    min_val:
        Minimum allowed element value (inclusive).
    max_val:
        Maximum allowed element value (inclusive).
    clamp:
        When ``True`` values are silently clamped instead of raising.
    metadata:
        Arbitrary key/value metadata stored alongside the tensor.
    """

    def __init__(
        self,
        tensor: Tensor,
        min_val: float | None = None,
        max_val: float | None = None,
        clamp: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tensor.dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {tensor.dtype!r}. "
                f"Allowed: {sorted(_ALLOWED_DTYPES)}"
            )

        self._tensor = self._validate(tensor, min_val, max_val, clamp)
        self.min_val  = min_val
        self.max_val  = max_val
        self.clamp    = clamp
        self.metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        t: Tensor,
        lo: float | None,
        hi: float | None,
        clamp: bool,
    ) -> Tensor:
        if lo is None and hi is None:
            return t
        data = list(t._data)
        for i, v in enumerate(data):
            if lo is not None and v < lo:
                if clamp:
                    data[i] = lo
                else:
                    raise ValueError(
                        f"Element {v} below min_val {lo} at flat index {i}"
                    )
            if hi is not None and v > hi:
                if clamp:
                    data[i] = hi
                else:
                    raise ValueError(
                        f"Element {v} above max_val {hi} at flat index {i}"
                    )
        return Tensor(t.dims, data, t.dtype, t.name)

    # ------------------------------------------------------------------
    # Delegated tensor access
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self) -> str:
        return self._tensor.dtype

    def get(self, *idx: int) -> float:
        return self._tensor.get(*idx)

    def set(self, *idx_and_val) -> None:
        *idx, val = idx_and_val
        if self.min_val is not None and val < self.min_val:
            if self.clamp:
                val = self.min_val
            else:
                raise ValueError(f"Value {val} below min_val {self.min_val}")
        if self.max_val is not None and val > self.max_val:
            if self.clamp:
                val = self.max_val
            else:
                raise ValueError(f"Value {val} above max_val {self.max_val}")
        self._tensor.set(*idx, val)

    def __repr__(self) -> str:
        return (
            f"SafeTensor(shape={self.shape}, dtype={self.dtype!r}, "
            f"min={self.min_val}, max={self.max_val})"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist this SafeTensor to *path* as JSON with CRC32 checksum."""
        payload = {
            "tensor":   self._tensor.to_dict(),
            "min_val":  self.min_val,
            "max_val":  self.max_val,
            "clamp":    self.clamp,
            "metadata": self.metadata,
        }
        raw = json.dumps(payload, separators=(",", ":"))
        crc = zlib.crc32(raw.encode()) & 0xFFFFFFFF
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"crc32": crc, "data": payload}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SafeTensor":
        """Load a SafeTensor from a JSON file, verifying the CRC32."""
        with open(path, encoding="utf-8") as fh:
            outer = json.load(fh)
        payload = outer["data"]
        raw = json.dumps(payload, separators=(",", ":"))
        expected_crc = zlib.crc32(raw.encode()) & 0xFFFFFFFF
        if outer.get("crc32") != expected_crc:
            raise ValueError(
                f"CRC32 mismatch: file may be corrupt "
                f"(expected {expected_crc:#010x}, got {outer.get('crc32'):#010x})"
            )
        tensor = Tensor.from_dict(payload["tensor"])
        return cls(
            tensor=tensor,
            min_val=payload.get("min_val"),
            max_val=payload.get("max_val"),
            clamp=payload.get("clamp", False),
            metadata=payload.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def zeros(
        cls, *dims: int, dtype: str = "float32", name: str = "",
        min_val: float | None = None, max_val: float | None = None,
    ) -> "SafeTensor":
        return cls(Tensor.zeros(*dims, dtype=dtype, name=name), min_val, max_val)

    @classmethod
    def ones(
        cls, *dims: int, dtype: str = "float32", name: str = "",
        min_val: float | None = None, max_val: float | None = None,
    ) -> "SafeTensor":
        return cls(Tensor.ones(*dims, dtype=dtype, name=name), min_val, max_val)

    @classmethod
    def from_tensor(
        cls, tensor: Tensor,
        min_val: float | None = None,
        max_val: float | None = None,
        clamp: bool = False,
    ) -> "SafeTensor":
        return cls(tensor, min_val, max_val, clamp)
