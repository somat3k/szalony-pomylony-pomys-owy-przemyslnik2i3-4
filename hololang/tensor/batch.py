"""SpreadTensor and BatchContainer — containerised tensor batching.

This module bridges the CROF message layer to the tensor compute layer.
Large tensors arriving via :class:`~hololang.crof.envelope.Envelope` payloads
are *spread* into fixed-size batches for efficient model deployment.

Design
------
``SpreadTensor``
    Takes a :class:`~hololang.tensor.tensor.Tensor` of shape ``(N, ...)``
    and partitions it along axis 0 into slices of at most *batch_size* rows.
    The final slice may be smaller than *batch_size* (the *remainder* batch).

``BatchContainer``
    Wraps the list of batches produced by :class:`SpreadTensor` together
    with envelope provenance, target model name, and execution metadata.
    Supports reassembly back to a single flat :class:`Tensor` via
    :meth:`BatchContainer.as_flat`.

``BatchResult``
    Lightweight summary returned after a container has been processed by a
    model deployment.  Carries per-batch latencies and the combined output
    tensor.

This module has **no** hard dependency on TensorFlow; it operates purely
with the built-in :class:`~hololang.tensor.tensor.Tensor` type.  The
``TensorComputeNode`` in :mod:`hololang.crof.compute` is responsible for
converting real TF tensors to/from this format.
"""

from __future__ import annotations

import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hololang.tensor.tensor import Tensor


# ---------------------------------------------------------------------------
# SpreadTensor
# ---------------------------------------------------------------------------

class SpreadTensor:
    """Partition a tensor along axis 0 into fixed-size batches.

    Parameters
    ----------
    tensor:
        Source tensor of shape ``(N, d1, d2, …)``.  *N* is the number of
        samples; the remaining dimensions are the feature dimensions.
    batch_size:
        Maximum number of samples per batch (must be ≥ 1).
    name:
        Optional name tag for debugging.

    Raises
    ------
    ValueError
        When *tensor* is scalar (0-d) or *batch_size* is < 1.
    """

    def __init__(
        self,
        tensor: Tensor,
        batch_size: int = 32,
        name: str = "",
    ) -> None:
        if not tensor.dims:
            raise ValueError("SpreadTensor requires at least a 1-D tensor")
        if batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")
        self.source     = tensor
        self.batch_size = batch_size
        self.name       = name or f"spread_{tensor.name or 'T'}"
        self._N:        int = tensor.dims[0]
        self._feat_dims: list[int] = tensor.dims[1:]  # may be empty for 1-D
        self._feat_size: int = 1
        for d in self._feat_dims:
            self._feat_size *= d

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Total number of samples (rows) in the source tensor."""
        return self._N

    @property
    def n_batches(self) -> int:
        """Number of batches produced by :meth:`spread`."""
        if self._N == 0:
            return 0
        return (self._N + self.batch_size - 1) // self.batch_size

    @property
    def remainder(self) -> int:
        """Number of samples in the last (possibly smaller) batch."""
        if self._N == 0:
            return 0
        r = self._N % self.batch_size
        return r if r != 0 else self.batch_size

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def spread(self) -> list[Tensor]:
        """Partition the source tensor and return a list of batch tensors.

        Each batch tensor has shape ``(b, d1, d2, …)`` where ``b ≤ batch_size``.
        For a 1-D source of shape ``(N,)`` the batches have shape ``(b,)``.

        Returns
        -------
        list[Tensor]
            Ordered list of batch tensors; the last batch may have fewer than
            *batch_size* rows.  Returns an empty list when *N* == 0.
        """
        src_data = self.source._data
        batches: list[Tensor] = []
        for start in range(0, self._N, self.batch_size):
            end = min(start + self.batch_size, self._N)
            b   = end - start
            # Slice the flat data for rows [start, end)
            flat_start = start * self._feat_size
            flat_end   = end   * self._feat_size
            batch_data = src_data[flat_start:flat_end]
            batch_dims = [b] + self._feat_dims
            batches.append(
                Tensor(
                    dims  = batch_dims,
                    data  = list(batch_data),
                    dtype = self.source.dtype,
                    name  = f"{self.name}_b{len(batches)}",
                )
            )
        return batches

    def __repr__(self) -> str:
        return (
            f"SpreadTensor(name={self.name!r}, N={self._N}, "
            f"batch_size={self.batch_size}, n_batches={self.n_batches})"
        )


# ---------------------------------------------------------------------------
# BatchContainer
# ---------------------------------------------------------------------------

@dataclass
class BatchContainer:
    """Containerised collection of tensor batches ready for model deployment.

    Created by :func:`from_spread` or :func:`from_envelope`; consumed by
    :class:`~hololang.crof.compute.ModelDeployment`.

    Attributes
    ----------
    container_id:
        Unique identifier for this batch container.
    batches:
        Ordered list of batch tensors (output of :meth:`SpreadTensor.spread`).
    model_target:
        Name of the model that should process these batches.
    provenance:
        Envelope IDs that contributed to this container.
    metadata:
        Free-form extension fields (feature names, dtype, etc.).
    created_at:
        Unix timestamp (float seconds) when the container was created.
    """
    container_id: str             = field(default_factory=lambda: str(uuid.uuid4()))
    batches:      list[Tensor]    = field(default_factory=list)
    model_target: str             = ""
    provenance:   list[str]       = field(default_factory=list)
    metadata:     dict[str, Any]  = field(default_factory=dict)
    created_at:   float           = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_batches(self) -> int:
        return len(self.batches)

    @property
    def total_items(self) -> int:
        """Total number of samples across all batches."""
        return sum(b.dims[0] for b in self.batches if b.dims)

    @property
    def batch_size(self) -> int:
        """Nominal batch size (from the first batch, 0 if empty)."""
        return self.batches[0].dims[0] if self.batches else 0

    @property
    def feature_dims(self) -> list[int]:
        """Feature dimensions (everything after axis 0 in the first batch)."""
        if not self.batches:
            return []
        return list(self.batches[0].dims[1:])

    # ------------------------------------------------------------------
    # Reassembly
    # ------------------------------------------------------------------

    def as_flat(self) -> Tensor:
        """Reassemble all batches into a single tensor of shape ``(N, ...)``.

        Returns an empty zero tensor of shape ``(0,)`` when the container
        holds no batches.
        """
        if not self.batches:
            return Tensor([0])
        first = self.batches[0]
        feat_dims = list(first.dims[1:])
        all_data: list[float] = []
        n_total = 0
        for batch in self.batches:
            all_data.extend(batch._data)
            n_total += batch.dims[0]
        return Tensor(
            dims  = [n_total] + feat_dims,
            data  = all_data,
            dtype = first.dtype,
            name  = f"reassembled_{self.container_id[:8]}",
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise container metadata (not the batch data) to a dict."""
        return {
            "container_id": self.container_id,
            "model_target": self.model_target,
            "provenance":   list(self.provenance),
            "n_batches":    self.n_batches,
            "total_items":  self.total_items,
            "feature_dims": self.feature_dims,
            "created_at":   self.created_at,
            "metadata":     dict(self.metadata),
        }

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_spread(
        cls,
        tensor: Tensor,
        batch_size: int = 32,
        model_target: str = "",
        provenance: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BatchContainer":
        """Spread *tensor* and wrap the batches in a :class:`BatchContainer`.

        Parameters
        ----------
        tensor:
            Source tensor of shape ``(N, ...)``.
        batch_size:
            Samples per batch.
        model_target:
            Name of the model that should process these batches.
        provenance:
            Envelope IDs to attach as provenance.
        metadata:
            Arbitrary key-value metadata.
        """
        spreader = SpreadTensor(tensor, batch_size=batch_size)
        batches  = spreader.spread()
        return cls(
            batches      = batches,
            model_target = model_target,
            provenance   = list(provenance or []),
            metadata     = dict(metadata or {}),
        )

    @classmethod
    def from_envelope_payload(
        cls,
        payload: bytes,
        batch_size: int = 32,
        model_target: str = "",
        envelope_id: str = "",
    ) -> "BatchContainer":
        """Decode a JSON-encoded tensor payload from an envelope and spread it.

        Expected payload JSON schema::

            {
                "dims":  [N, d1, d2, ...],
                "data":  [float, ...],
                "dtype": "float32"        // optional
            }

        When the payload cannot be decoded (e.g. raw bytes), a 1-D tensor
        of the raw bytes' float32 values is created.
        """
        try:
            obj  = json.loads(payload.decode("utf-8", errors="replace"))
            dims = list(obj.get("dims", [len(obj.get("data", []))]))
            data = [float(x) for x in obj.get("data", [])]
            dtype = obj.get("dtype", "float32")
        except Exception:  # noqa: BLE001
            # Fall back: interpret raw bytes as packed float32
            n = len(payload) // 4
            if n > 0:
                data  = list(struct.unpack(f"<{n}f", payload[:n * 4]))
                dims  = [n]
                dtype = "float32"
            else:
                data = []
                dims = []
                dtype = "float32"

        if not dims or not data:
            # Empty payload → single zero-batch
            return cls(
                batches      = [Tensor([1], [0.0])],
                model_target = model_target,
                provenance   = [envelope_id] if envelope_id else [],
            )

        tensor = Tensor(dims=dims, data=data, dtype=dtype)
        return cls.from_spread(
            tensor       = tensor,
            batch_size   = batch_size,
            model_target = model_target,
            provenance   = [envelope_id] if envelope_id else [],
        )

    def __repr__(self) -> str:
        return (
            f"BatchContainer(id={self.container_id[:8]}…, "
            f"batches={self.n_batches}, total_items={self.total_items}, "
            f"model={self.model_target!r})"
        )
