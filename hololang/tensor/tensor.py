"""N-dimensional tensor with element-wise operations.

A pure-Python implementation that works without NumPy.
When NumPy is available it is used transparently for performance.
"""

from __future__ import annotations

import math
import itertools
from typing import Any, Callable, Iterable, Iterator

from hololang.tensor.helpers import strides as _strides_fn
from hololang.tensor.helpers import flat_index as _flat_index_fn
from hololang.tensor.helpers import total_elements as _total_fn


# ---------------------------------------------------------------------------
# Private aliases kept for internal use (backwards-compatible)
# ---------------------------------------------------------------------------

def _strides(dims: list[int]) -> list[int]:
    return _strides_fn(dims)


def _flat_index(multi_idx: tuple[int, ...], strides: list[int]) -> int:
    return _flat_index_fn(multi_idx, strides)


def _total(dims: list[int]) -> int:
    return _total_fn(dims)


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """N-dimensional dense tensor backed by a flat list.

    Parameters
    ----------
    dims:
        Shape, e.g. ``[4, 4]`` for a 4×4 matrix.
    data:
        Initial flat data.  If *None* all elements are zero.
    dtype:
        String type tag (``"float32"``, ``"float64"``, ``"int32"`` …).
    name:
        Human-readable name for debugging.
    """

    def __init__(
        self,
        dims: list[int],
        data: list[float] | None = None,
        dtype: str = "float32",
        name: str = "",
    ) -> None:
        self.dims:  list[int] = list(dims)
        self.dtype: str = dtype
        self.name:  str = name
        self.meta:  dict[str, Any] = {}

        total = _total(dims) if dims else 1
        if data is not None:
            if len(data) != total:
                raise ValueError(
                    f"Data length {len(data)} doesn't match shape {dims} (total={total})"
                )
            self._data: list[float] = list(data)
        else:
            self._data = [0.0] * total

        self._strides = _strides(dims)

    # ------------------------------------------------------------------
    # Shape / info
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dims)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def size(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"Tensor(name={self.name!r}, shape={self.shape}, "
            f"dtype={self.dtype!r})"
        )

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def _resolve(self, idx: tuple[int, ...]) -> int:
        return _flat_index(idx, self._strides)

    def get(self, *idx: int) -> float:
        return self._data[self._resolve(idx)]

    def set(self, *idx_and_val) -> None:
        *idx, val = idx_and_val
        self._data[self._resolve(tuple(idx))] = float(val)

    def __getitem__(self, idx) -> float | "Tensor":
        if isinstance(idx, tuple):
            if len(idx) == self.ndim:
                return self._data[self._resolve(idx)]
            # Partial indexing – return slice
            return self._slice(idx)
        if self.ndim == 1:
            return self._data[idx]
        return self._slice((idx,))

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, tuple):
            self._data[self._resolve(idx)] = float(value)
        elif self.ndim == 1:
            self._data[idx] = float(value)
        else:
            raise IndexError("Use a full tuple index to set values")

    def _slice(self, partial: tuple) -> "Tensor":
        """Return a lower-dimensional sub-tensor."""
        new_dims = self.dims[len(partial):]
        new_strides = self._strides[len(partial):]
        offset = _flat_index(partial, self._strides[:len(partial)])
        new_size = _total(new_dims) if new_dims else 1
        new_data = self._data[offset: offset + new_size]
        return Tensor(dims=new_dims, data=new_data, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Element-wise ops
    # ------------------------------------------------------------------

    def _apply(self, other: "Tensor | float | int", fn: Callable) -> "Tensor":
        if isinstance(other, (int, float)):
            return Tensor(
                dims=self.dims,
                data=[fn(x, other) for x in self._data],
                dtype=self.dtype,
            )
        if isinstance(other, Tensor):
            if other.shape != self.shape:
                raise ValueError(
                    f"Shape mismatch: {self.shape} vs {other.shape}"
                )
            return Tensor(
                dims=self.dims,
                data=[fn(a, b) for a, b in zip(self._data, other._data)],
                dtype=self.dtype,
            )
        return NotImplemented

    def __add__(self, other):  return self._apply(other, lambda a, b: a + b)
    def __sub__(self, other):  return self._apply(other, lambda a, b: a - b)
    def __mul__(self, other):  return self._apply(other, lambda a, b: a * b)
    def __truediv__(self, other): return self._apply(other, lambda a, b: a / b)
    def __neg__(self):         return Tensor(self.dims, [-x for x in self._data], self.dtype)

    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self) -> float:
        return sum(self._data)

    def mean(self) -> float:
        return sum(self._data) / self.size

    def min(self) -> float:
        return min(self._data)

    def max(self) -> float:
        return max(self._data)

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self._data))

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def reshape(self, *new_dims: int) -> "Tensor":
        new_total = _total(list(new_dims))
        if new_total != self.size:
            raise ValueError(
                f"Cannot reshape size {self.size} to {new_dims}"
            )
        return Tensor(list(new_dims), list(self._data), self.dtype)

    def flatten(self) -> "Tensor":
        return Tensor([self.size], list(self._data), self.dtype)

    def transpose(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("transpose() only supports 2-D tensors")
        r, c = self.dims
        new_data = [self._data[i * c + j] for j in range(c) for i in range(r)]
        return Tensor([c, r], new_data, self.dtype)

    def matmul(self, other: "Tensor") -> "Tensor":
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul() only supports 2-D tensors")
        m, k = self.dims
        k2, n = other.dims
        if k != k2:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} @ {other.shape}")
        result = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += self._data[i * k + p] * other._data[p * n + j]
                result[i * n + j] = s
        return Tensor([m, n], result, self.dtype)

    def apply_fn(self, fn: Callable[[float], float]) -> "Tensor":
        return Tensor(self.dims, [fn(x) for x in self._data], self.dtype)

    def clip(self, lo: float, hi: float) -> "Tensor":
        return Tensor(self.dims, [max(lo, min(hi, x)) for x in self._data], self.dtype)

    def normalize(self) -> "Tensor":
        n = self.norm()
        if n == 0:
            return Tensor(self.dims, list(self._data), self.dtype)
        return self / n

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def zeros(cls, *dims: int, dtype: str = "float32", name: str = "") -> "Tensor":
        return cls(list(dims), dtype=dtype, name=name)

    @classmethod
    def ones(cls, *dims: int, dtype: str = "float32", name: str = "") -> "Tensor":
        total = _total(list(dims))
        return cls(list(dims), [1.0] * total, dtype=dtype, name=name)

    @classmethod
    def eye(cls, n: int, dtype: str = "float32") -> "Tensor":
        data = [1.0 if i == j else 0.0
                for i in range(n) for j in range(n)]
        return cls([n, n], data, dtype)

    @classmethod
    def from_nested(cls, nested: list, dtype: str = "float32",
                    name: str = "") -> "Tensor":
        """Build a tensor from a nested list."""
        def _shape(arr):
            if not isinstance(arr, list):
                return []
            return [len(arr)] + _shape(arr[0])

        def _flatten(arr):
            if not isinstance(arr, list):
                return [float(arr)]
            result = []
            for sub in arr:
                result.extend(_flatten(sub))
            return result

        dims = _shape(nested)
        data = _flatten(nested)
        return cls(dims, data, dtype, name)

    # ------------------------------------------------------------------
    # Serialisation (safetensor-compatible dict)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name":  self.name,
            "dims":  self.dims,
            "dtype": self.dtype,
            "data":  list(self._data),
            "meta":  self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tensor":
        t = cls(d["dims"], d["data"], d["dtype"], d.get("name", ""))
        t.meta = d.get("meta", {})
        return t

    # ------------------------------------------------------------------
    # Python sequence helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[float]:
        if self.ndim == 1:
            return iter(self._data)
        # Iterate over first dimension
        return (self._slice((i,)) for i in range(self.dims[0]))

    def __len__(self) -> int:
        return self.dims[0] if self.dims else 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.dims == other.dims and self._data == other._data
