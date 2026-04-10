"""Tensor Compute Node — Class-5 node with TF backend, model deployment, and receiver routing.

This module implements the **Transformation Plane** (Plane E) compute tier
that sits at the intersection of:

* The **CROF message layer** (:class:`~hololang.crof.envelope.Envelope`)
* The **Receiver aspect system** (:class:`~hololang.crof.receiver.ReceiverChain`)
* The **SpreadTensor / BatchContainer** batching layer (:mod:`hololang.tensor.batch`)
* The **GAS embedding memory** (:class:`~hololang.tensor.embeddings.EmbeddingSpace`)
* The **Kernel32 extended VM** (:class:`~hololang.vm.kernel32.Kernel32`)

Architecture
------------

::

    Envelope (CROF msg)
         │
         ▼
    ReceiverChain ──► ReceiverDecision
    (aspect routing)        │
                            │  target_layer == "model"
                            ▼
                    SpreadTensor ──► BatchContainer
                                           │
                                           ▼
                                   ModelDeployment
                                   (TFBackend.predict)
                                           │
                                           ▼
                                    ModelOutput  ──► TransformResult
                                                      (provenance, confidence)

Backend abstraction
-------------------
``TFBackend``
    Protocol class that every backend must satisfy.  Backends are
    interchangeable at runtime — the node calls only
    ``backend.predict(batch, model_name) -> Tensor``.

``MockTFBackend``
    Pure-Python backend with no external dependencies.  Used in tests and
    as a drop-in when TensorFlow is not installed.  Its prediction function
    is configurable so tests can inject custom logic.

``TensorFlowBackend``
    Real TensorFlow 2.x backend (lazy import).  Raises
    :class:`ImportError` with a helpful message when ``tensorflow`` is not
    installed.  Loads and caches ``tf.saved_model`` / ``tf.keras.Model``
    objects.

``TensorComputeNode``
    A :class:`~hololang.crof.transform.TransformModule` subclass wired to a
    :class:`~hololang.crof.receiver.ReceiverChain` and a registry of
    :class:`ModelDeployment` objects.  It overrides :meth:`execute` to
    perform the full envelope → batch → inference → result pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from hololang.tensor.tensor import Tensor
from hololang.tensor.batch import BatchContainer, SpreadTensor
from hololang.tensor.embeddings import EmbeddingSpace
from hololang.crof.envelope import Envelope
from hololang.crof.receiver import ReceiverChain, ReceiverDecision
from hololang.crof.transform import TransformModule, TransformResult


# ---------------------------------------------------------------------------
# TFBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TFBackend(Protocol):
    """Protocol that every tensor-compute backend must satisfy.

    A backend is responsible for:

    1. Accepting a batch :class:`~hololang.tensor.tensor.Tensor` of shape
       ``(B, d1, d2, …)``
    2. Running inference / computation for a named *model_name*
    3. Returning a result :class:`~hololang.tensor.tensor.Tensor`

    The protocol is intentionally minimal so that wrappers for TensorFlow,
    PyTorch, ONNX, scikit-learn, or any other framework can be plugged in
    without changing the :class:`TensorComputeNode`.
    """

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    def predict(self, batch: Tensor, model_name: str) -> Tensor:
        """Run inference on a single batch tensor.

        Parameters
        ----------
        batch:
            Input tensor of shape ``(B, ...)``.
        model_name:
            The model to invoke (the backend resolves this internally).

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, ...)``.
        """
        ...

    def is_available(self) -> bool:
        """Return ``True`` when the backend runtime is installed and ready."""
        ...


# ---------------------------------------------------------------------------
# MockTFBackend
# ---------------------------------------------------------------------------

class MockTFBackend:
    """Pure-Python mock backend — no TensorFlow required.

    By default :meth:`predict` returns the input batch unchanged (identity
    transform).  You can override this by supplying a custom *predict_fn*
    that maps ``(batch, model_name) -> Tensor``.

    Parameters
    ----------
    predict_fn:
        Optional callable ``(batch: Tensor, model_name: str) -> Tensor``.
        When ``None`` the identity function is used.
    backend_name:
        Name reported by the :attr:`name` property.
    """

    def __init__(
        self,
        predict_fn: Callable[[Tensor, str], Tensor] | None = None,
        backend_name: str = "mock",
    ) -> None:
        self._predict_fn   = predict_fn
        self._backend_name = backend_name
        self._call_count   = 0

    @property
    def name(self) -> str:
        return self._backend_name

    def predict(self, batch: Tensor, model_name: str) -> Tensor:
        self._call_count += 1
        if self._predict_fn is not None:
            return self._predict_fn(batch, model_name)
        # Identity: return the batch unchanged
        return Tensor(
            dims  = list(batch.dims),
            data  = list(batch._data),
            dtype = batch.dtype,
            name  = f"{model_name}_out",
        )

    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

    def __repr__(self) -> str:
        return f"MockTFBackend(name={self._backend_name!r}, calls={self._call_count})"


# ---------------------------------------------------------------------------
# TensorFlowBackend
# ---------------------------------------------------------------------------

class TensorFlowBackend:
    """TensorFlow 2.x backend with SavedModel / Keras model loading.

    This backend lazy-imports TensorFlow so the module can be imported
    even when TF is not installed — only :meth:`predict` and
    :meth:`is_available` are affected.

    Parameters
    ----------
    model_dir:
        Filesystem path to a ``tf.saved_model`` or Keras ``.h5`` file.
        When provided the model is loaded on first use.  Models can also
        be registered programmatically via :meth:`register_model`.
    backend_name:
        Name reported by the :attr:`name` property.
    """

    def __init__(
        self,
        model_dir: str = "",
        backend_name: str = "tensorflow",
    ) -> None:
        self._backend_name = backend_name
        self._model_dir    = model_dir
        self._models: dict[str, Any] = {}   # model_name -> tf.Module
        self._tf = None   # lazily imported

    @property
    def name(self) -> str:
        return self._backend_name

    def _import_tf(self) -> Any:
        if self._tf is None:
            try:
                import tensorflow as tf  # noqa: PLC0415
                self._tf = tf
            except ImportError as exc:
                raise ImportError(
                    "TensorFlowBackend requires TensorFlow 2.x. "
                    "Install it with: pip install tensorflow"
                ) from exc
        return self._tf

    def is_available(self) -> bool:
        """Return ``True`` when TensorFlow can be imported."""
        try:
            self._import_tf()
            return True
        except ImportError:
            return False

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a pre-loaded TF model under *model_name*."""
        self._models[model_name] = model

    def load_model(self, model_name: str, path: str) -> None:
        """Load a TF SavedModel or Keras model from *path*."""
        tf = self._import_tf()
        try:
            model = tf.saved_model.load(path)
        except Exception:  # noqa: BLE001
            model = tf.keras.models.load_model(path)
        self._models[model_name] = model

    def predict(self, batch: Tensor, model_name: str) -> Tensor:
        """Run TF inference on *batch* using the named model.

        The batch is converted to a ``tf.Tensor``, passed through the model's
        ``__call__`` (or ``predict``) method, and the result is converted back
        to a CROF :class:`~hololang.tensor.tensor.Tensor`.
        """
        tf = self._import_tf()
        model = self._models.get(model_name)
        if model is None:
            raise KeyError(
                f"TensorFlowBackend: no model registered as {model_name!r}. "
                "Use register_model() or load_model() first."
            )
        # Build TF tensor from batch
        import numpy as _np  # noqa: PLC0415
        np_data = _np.array(batch._data, dtype=_np.float32).reshape(batch.dims)
        tf_in   = tf.constant(np_data)
        # Try __call__ first (SavedModel), fall back to .predict (Keras)
        try:
            tf_out = model(tf_in, training=False)
        except TypeError:
            tf_out = model.predict(np_data)
        # Convert back
        out_np   = _np.array(tf_out).astype(_np.float32)
        out_flat = out_np.flatten().tolist()
        out_dims = list(out_np.shape)
        return Tensor(
            dims  = out_dims,
            data  = out_flat,
            dtype = "float32",
            name  = f"{model_name}_tf_out",
        )

    def __repr__(self) -> str:
        available = self.is_available()
        return (
            f"TensorFlowBackend(name={self._backend_name!r}, "
            f"models={list(self._models)}, available={available})"
        )


# ---------------------------------------------------------------------------
# ModelOutput
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """Result of a :class:`ModelDeployment` inference run.

    Attributes
    ----------
    deployment_name:
        Name of the model deployment that produced this output.
    model_name:
        Name of the specific model/variant used.
    predictions:
        One output tensor per input batch.
    latencies_ms:
        Wall-clock inference time for each batch in milliseconds.
    provenance:
        Envelope IDs that contributed to the input container.
    confidence:
        Aggregate confidence score [0, 1].  For ``MockTFBackend`` this is
        always ``1.0``; real backends should compute it from output
        distributions.
    error:
        Non-empty when inference raised an exception.
    metadata:
        Extension fields (batch count, backend name, etc.).
    """
    deployment_name: str            = ""
    model_name:      str            = ""
    predictions:     list[Tensor]   = field(default_factory=list)
    latencies_ms:    list[float]    = field(default_factory=list)
    provenance:      list[str]      = field(default_factory=list)
    confidence:      float          = 1.0
    error:           str            = ""
    metadata:        dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def n_batches(self) -> int:
        return len(self.predictions)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latencies_ms)

    @property
    def avg_latency_ms(self) -> float:
        n = len(self.latencies_ms)
        return (self.total_latency_ms / n) if n else 0.0

    def as_flat_prediction(self) -> Tensor:
        """Concatenate all per-batch predictions into one tensor.

        Returns a zero-dim tensor when there are no predictions.
        """
        if not self.predictions:
            return Tensor([0])
        first = self.predictions[0]
        feat_dims = list(first.dims[1:])
        all_data: list[float] = []
        n_total = 0
        for pred in self.predictions:
            all_data.extend(pred._data)
            n_total += pred.dims[0]
        return Tensor(
            dims  = [n_total] + feat_dims,
            data  = all_data,
            dtype = first.dtype,
            name  = f"{self.deployment_name}_flat",
        )

    def __repr__(self) -> str:
        status = "OK" if self.ok else f"ERR:{self.error}"
        return (
            f"ModelOutput(deployment={self.deployment_name!r}, "
            f"batches={self.n_batches}, "
            f"total_ms={self.total_latency_ms:.1f}, {status})"
        )


# ---------------------------------------------------------------------------
# ModelDeployment
# ---------------------------------------------------------------------------

class ModelDeployment:
    """A deployed model that accepts :class:`~hololang.tensor.batch.BatchContainer`
    inputs and produces :class:`ModelOutput` results.

    Parameters
    ----------
    name:
        Unique deployment name (e.g. ``"arbitrage-scorer-v2"``).
    model_name:
        Name of the model to invoke on the backend.
    backend:
        The :class:`TFBackend`-compatible backend that runs inference.
    confidence_fn:
        Optional callable ``(output: Tensor) -> float`` to compute a
        confidence score from an output tensor.  Defaults to ``1.0``.
    max_batch_size:
        Maximum batch size the deployment accepts.  When a batch exceeds
        this size it is re-split before inference.  ``0`` = no limit.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        backend: TFBackend | MockTFBackend,
        confidence_fn: Callable[[Tensor], float] | None = None,
        max_batch_size: int = 0,
    ) -> None:
        self.name           = name
        self.model_name     = model_name
        self.backend        = backend
        self.confidence_fn  = confidence_fn
        self.max_batch_size = max_batch_size
        self._run_count:    int   = 0
        self._total_ms:     float = 0.0

    def _split_if_needed(self, batch: Tensor) -> list[Tensor]:
        """Re-split a batch if it exceeds :attr:`max_batch_size`."""
        if self.max_batch_size > 0 and batch.dims and batch.dims[0] > self.max_batch_size:
            spreader = SpreadTensor(batch, batch_size=self.max_batch_size)
            return spreader.spread()
        return [batch]

    def run(self, container: BatchContainer) -> ModelOutput:
        """Run inference on every batch in *container*.

        Parameters
        ----------
        container:
            Input batches and provenance from the spread-tensor layer.

        Returns
        -------
        ModelOutput
            Collected per-batch predictions, latencies, and metadata.
        """
        predictions:  list[Tensor] = []
        latencies_ms: list[float]  = []
        error = ""

        for batch in container.batches:
            sub_batches = self._split_if_needed(batch)
            for sb in sub_batches:
                t0 = time.perf_counter()
                try:
                    pred = self.backend.predict(sb, self.model_name)
                    predictions.append(pred)
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    # Push zero-prediction so indices stay aligned
                    predictions.append(Tensor(dims=list(sb.dims), data=[0.0] * sb.size))
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Aggregate confidence
        if predictions and not error:
            if self.confidence_fn is not None:
                scores = [self.confidence_fn(p) for p in predictions]
                confidence = sum(scores) / len(scores)
            else:
                confidence = 1.0
        else:
            confidence = 0.0 if error else 1.0

        total_ms = sum(latencies_ms)
        self._run_count  += 1
        self._total_ms   += total_ms

        return ModelOutput(
            deployment_name = self.name,
            model_name      = self.model_name,
            predictions     = predictions,
            latencies_ms    = latencies_ms,
            provenance      = list(container.provenance),
            confidence      = confidence,
            error           = error,
            metadata        = {
                "backend":     self.backend.name,
                "n_batches":   len(container.batches),
                "total_items": container.total_items,
            },
        )

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def avg_run_ms(self) -> float:
        return (self._total_ms / self._run_count) if self._run_count else 0.0

    def __repr__(self) -> str:
        return (
            f"ModelDeployment(name={self.name!r}, "
            f"model={self.model_name!r}, "
            f"backend={self.backend.name!r}, "
            f"runs={self._run_count})"
        )


# ---------------------------------------------------------------------------
# TensorComputeNode  (Class-5 Tensor Compute Node)
# ---------------------------------------------------------------------------

class TensorComputeNode(TransformModule):
    """Class-5 Tensor Compute Node — routes envelopes via ReceiverChain to
    ModelDeployments, SpreadTensor batching, and optional GAS embeddings.

    This is the primary integration point between the CROF message layer
    and the tensor compute layer.

    Execution flow for each :meth:`execute` call:

    1. :class:`~hololang.crof.receiver.ReceiverChain` resolves a
       :class:`~hololang.crof.receiver.ReceiverDecision` from the envelope.
    2. When ``target_layer == "model"``:
       a. The envelope payload is decoded to a
          :class:`~hololang.tensor.batch.BatchContainer` via
          :meth:`~hololang.tensor.batch.BatchContainer.from_envelope_payload`.
       b. The named :class:`ModelDeployment` is invoked.
       c. A :class:`~hololang.crof.transform.TransformResult` is returned with
          the :class:`ModelOutput` as ``output``.
    3. When ``target_layer == "embedding"``:
       a. The payload is decoded to a vector.
       b. The named :class:`~hololang.tensor.embeddings.EmbeddingSpace` is
          queried (``most_similar``).
       c. Results are returned in the ``TransformResult``.
    4. When ``target_layer == "tensor"``:
       a. The payload is decoded and spread into a :class:`BatchContainer`.
       b. No model is invoked; the container is returned as-is.
    5. Unmatched envelopes (``passthrough``) return a no-op result.

    Parameters
    ----------
    module_id:
        Unique module identifier.
    chain:
        :class:`~hololang.crof.receiver.ReceiverChain` used for routing.
    deployments:
        Registry of :class:`ModelDeployment` objects keyed by name.
    embedding_spaces:
        Registry of :class:`~hololang.tensor.embeddings.EmbeddingSpace`
        objects keyed by name.
    default_batch_size:
        Fallback batch size when the receiver decision does not specify one.
    """

    def __init__(
        self,
        module_id: str = "",
        chain: ReceiverChain | None = None,
        deployments: dict[str, ModelDeployment] | None = None,
        embedding_spaces: dict[str, EmbeddingSpace] | None = None,
        default_batch_size: int = 32,
    ) -> None:
        super().__init__(
            module_id   = module_id or f"tcn-{str(uuid.uuid4())[:8]}",
            trust_class = "adaptive",
            description = "Class-5 Tensor Compute Node with receiver routing",
        )
        self.chain              = chain or ReceiverChain(name=f"{self.module_id}.chain")
        self._deployments:      dict[str, ModelDeployment]  = dict(deployments or {})
        self._embedding_spaces: dict[str, EmbeddingSpace]   = dict(embedding_spaces or {})
        self.default_batch_size = default_batch_size

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def register_deployment(self, deployment: ModelDeployment) -> None:
        """Register a :class:`ModelDeployment` under its :attr:`~ModelDeployment.name`."""
        self._deployments[deployment.name] = deployment

    def register_embedding_space(self, space: EmbeddingSpace) -> None:
        """Register a :class:`~hololang.tensor.embeddings.EmbeddingSpace`."""
        self._embedding_spaces[space.name] = space

    # ------------------------------------------------------------------
    # execute — override of TransformModule.execute
    # ------------------------------------------------------------------

    def execute(
        self,
        envelope: Envelope,
        context: dict[str, Any] | None = None,
    ) -> TransformResult:
        """Route *envelope* via the receiver chain and execute the appropriate layer.

        Returns
        -------
        TransformResult
            ``output`` is one of:

            * :class:`ModelOutput` — when routed to a model
            * :class:`~hololang.tensor.batch.BatchContainer` — when routed to tensor layer
            * ``list[tuple[str, float]]`` — when routed to embedding (similarity results)
            * ``None`` — passthrough / no-op
        """
        ctx = context or {}
        t0  = time.perf_counter()
        error = ""
        output: Any = None

        decision = self.chain.resolve(envelope)
        ctx["receiver_decision"] = decision

        try:
            if decision.routes_to_model:
                output = self._run_model(envelope, decision)
            elif decision.routes_to_embedding:
                output = self._run_embedding(envelope, decision)
            elif decision.routes_to_tensor:
                output = self._run_tensor(envelope, decision)
            # else: passthrough → output stays None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.invocations += 1
        self.total_ms    += dt_ms

        confidence = (
            output.confidence if isinstance(output, ModelOutput) and output.ok else
            (0.0 if error else 0.95)
        )

        return TransformResult(
            module_id   = self.module_id,
            output      = output,
            confidence  = confidence,
            provenance  = [envelope.envelope_id],
            duration_ms = dt_ms,
            error       = error,
            metadata    = {
                "decision":     repr(decision),
                "target_layer": decision.target_layer,
                "model_name":   decision.model_name,
            },
        )

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _run_model(
        self, envelope: Envelope, decision: ReceiverDecision
    ) -> ModelOutput:
        deployment = self._deployments.get(decision.model_name)
        if deployment is None:
            raise KeyError(
                f"TensorComputeNode: no deployment named {decision.model_name!r}. "
                "Use register_deployment() first."
            )
        container = BatchContainer.from_envelope_payload(
            payload      = envelope.payload,
            batch_size   = decision.batch_size,
            model_target = decision.model_name,
            envelope_id  = envelope.envelope_id,
        )
        return deployment.run(container)

    def _run_embedding(
        self, envelope: Envelope, decision: ReceiverDecision
    ) -> list[tuple[str, float]]:
        space_name = decision.model_name or next(
            iter(self._embedding_spaces), None
        )
        if space_name is None:
            raise KeyError("TensorComputeNode: no embedding space available")
        space = self._embedding_spaces.get(space_name)
        if space is None:
            raise KeyError(
                f"TensorComputeNode: no embedding space named {space_name!r}"
            )
        # Decode payload as a query vector
        try:
            obj   = json.loads(envelope.payload.decode("utf-8", errors="replace"))
            query = [float(x) for x in obj.get("vector", obj.get("data", []))]
        except Exception:  # noqa: BLE001
            import struct as _struct
            n     = max(1, len(envelope.payload) // 4)
            query = list(_struct.unpack(f"<{n}f", envelope.payload[:n * 4]))
        return space.most_similar(query, top_k=decision.batch_size)

    def _run_tensor(
        self, envelope: Envelope, decision: ReceiverDecision
    ) -> BatchContainer:
        return BatchContainer.from_envelope_payload(
            payload      = envelope.payload,
            batch_size   = decision.batch_size,
            model_target = decision.model_name,
            envelope_id  = envelope.envelope_id,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        module_id: str = "",
        default_batch_size: int = 32,
        backend: MockTFBackend | TensorFlowBackend | None = None,
    ) -> "TensorComputeNode":
        """Create a :class:`TensorComputeNode` with a fresh chain and a
        :class:`MockTFBackend` (or the provided *backend*).

        Quick-start::

            node = TensorComputeNode.build(module_id="scorer")
            node.chain.add(Receiver(
                name="score-receiver",
                predicate=lambda env: env.module_id == "score",
                target_layer="model",
                model_name="my-model",
            ))
            node.register_deployment(ModelDeployment(
                name="my-model",
                model_name="my-model",
                backend=MockTFBackend(),
            ))
        """
        _backend = backend or MockTFBackend()
        return cls(
            module_id           = module_id,
            default_batch_size  = default_batch_size,
        )

    def __repr__(self) -> str:
        return (
            f"TensorComputeNode(id={self.module_id!r}, "
            f"deployments={list(self._deployments)}, "
            f"invocations={self.invocations})"
        )
