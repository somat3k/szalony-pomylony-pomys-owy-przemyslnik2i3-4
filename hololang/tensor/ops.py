"""Standalone tensor activation and transformation functions.

All functions accept a :class:`~hololang.tensor.tensor.Tensor` and return a
new :class:`~hololang.tensor.tensor.Tensor` without modifying the input.
These are designed to be composed inside a
:class:`~hololang.tensor.transforms.TransformChain` or wired directly into a
:class:`~hololang.tensor.graph.ComputationGraph` node.

CPU execution — no external dependencies required.
"""

from __future__ import annotations

import math
from typing import Callable

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(t: Tensor) -> Tensor:
    """Rectified Linear Unit: ``max(0, x)`` element-wise."""
    return t.apply_fn(lambda v: max(0.0, v))


def leaky_relu(t: Tensor, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU: ``x if x >= 0 else alpha * x`` element-wise."""
    return t.apply_fn(lambda v: v if v >= 0.0 else alpha * v)


def sigmoid(t: Tensor) -> Tensor:
    """Sigmoid activation: ``1 / (1 + exp(-x))`` element-wise."""
    return t.apply_fn(lambda v: 1.0 / (1.0 + math.exp(-v)))


def tanh_act(t: Tensor) -> Tensor:
    """Hyperbolic tangent activation element-wise."""
    return t.apply_fn(math.tanh)


def elu(t: Tensor, alpha: float = 1.0) -> Tensor:
    """Exponential Linear Unit element-wise."""
    return t.apply_fn(
        lambda v: v if v >= 0.0 else alpha * (math.exp(v) - 1.0)
    )


def softmax(t: Tensor) -> Tensor:
    """Softmax over the entire (flattened) tensor.

    Returns a tensor of the same shape whose elements sum to 1.
    """
    data = t._data
    max_v = max(data)
    exps = [math.exp(v - max_v) for v in data]
    total = sum(exps)
    return Tensor(t.dims, [e / total for e in exps], t.dtype, t.name)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(t: Tensor) -> Tensor:
    """L2-normalise the tensor so its Euclidean norm equals 1."""
    return t.normalize()


def layer_norm(t: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation: subtract mean and divide by std over all elements."""
    data = t._data
    n = len(data)
    mean = sum(data) / n
    var = sum((v - mean) ** 2 for v in data) / n
    std = math.sqrt(var + eps)
    return Tensor(t.dims, [(v - mean) / std for v in data], t.dtype, t.name)


def batch_norm(t: Tensor, eps: float = 1e-5) -> Tensor:
    """Batch normalisation (treats the full tensor as a single batch)."""
    return layer_norm(t, eps)


def min_max_scale(t: Tensor, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """Scale tensor values linearly into ``[lo, hi]``."""
    data = t._data
    t_min = min(data)
    t_max = max(data)
    span = t_max - t_min
    if span == 0.0:
        return Tensor(t.dims, [lo] * len(data), t.dtype, t.name)
    return Tensor(
        t.dims,
        [lo + (v - t_min) / span * (hi - lo) for v in data],
        t.dtype,
        t.name,
    )


# ---------------------------------------------------------------------------
# Dropout (inference-time — identity; training uses mask)
# ---------------------------------------------------------------------------

def dropout(t: Tensor, rate: float = 0.5, seed: int | None = None) -> Tensor:
    """Apply inverted dropout mask.

    At inference ``rate=0`` behaves as identity.  During training each
    element is zeroed with probability *rate* and scaled by ``1/(1-rate)``.
    """
    if rate <= 0.0:
        return Tensor(t.dims, list(t._data), t.dtype, t.name)
    import random
    rng = random.Random(seed)
    scale = 1.0 / (1.0 - rate)
    data = [
        0.0 if rng.random() < rate else v * scale
        for v in t._data
    ]
    return Tensor(t.dims, data, t.dtype, t.name)


# ---------------------------------------------------------------------------
# Clipping & scaling
# ---------------------------------------------------------------------------

def clip(t: Tensor, lo: float, hi: float) -> Tensor:
    """Clamp all values to ``[lo, hi]``."""
    return t.clip(lo, hi)


def scale(t: Tensor, factor: float) -> Tensor:
    """Multiply every element by *factor*."""
    return t * factor


def offset(t: Tensor, bias: float) -> Tensor:
    """Add *bias* to every element."""
    return t + bias


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def apply(t: Tensor, fn: Callable[[float], float]) -> Tensor:
    """Apply an arbitrary scalar function element-wise."""
    return t.apply_fn(fn)
