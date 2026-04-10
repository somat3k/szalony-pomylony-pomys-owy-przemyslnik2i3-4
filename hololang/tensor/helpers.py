"""Low-level index helpers for N-dimensional tensor addressing.

These utilities are used internally by :mod:`hololang.tensor.tensor` and any
module that needs to compute flat offsets from multi-dimensional indices.
"""

from __future__ import annotations


def strides(dims: list[int]) -> list[int]:
    """Compute row-major (C-order) strides for *dims*.

    Parameters
    ----------
    dims:
        Shape of the tensor, e.g. ``[4, 3]``.

    Returns
    -------
    list[int]
        Stride for each dimension so that
        ``flat_index = sum(idx[i] * stride[i] for i)``.
    """
    s = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        s[i] = s[i + 1] * dims[i + 1]
    return s


def flat_index(multi_idx: tuple[int, ...], strides_: list[int]) -> int:
    """Convert a multi-dimensional index to a flat (1-D) position.

    Parameters
    ----------
    multi_idx:
        Per-dimension indices.
    strides_:
        Pre-computed strides (see :func:`strides`).
    """
    return sum(i * s for i, s in zip(multi_idx, strides_))


def total_elements(dims: list[int]) -> int:
    """Return the total number of elements in a tensor of shape *dims*."""
    result = 1
    for d in dims:
        result *= d
    return result
