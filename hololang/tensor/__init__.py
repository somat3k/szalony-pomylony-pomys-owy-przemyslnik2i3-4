"""hololang.tensor package – tensor processing subsystem."""
from hololang.tensor.helpers import strides, flat_index, total_elements
from hololang.tensor.tensor import Tensor
from hololang.tensor.safetensor import SafeTensor
from hololang.tensor.graph_node import GraphNode
from hololang.tensor.graph import ComputationGraph
from hololang.tensor.pool import TensorPool
from hololang.tensor.ops import (
    relu, leaky_relu, sigmoid, tanh_act, elu, softmax,
    normalize, layer_norm, batch_norm, min_max_scale,
    dropout, clip, scale, offset, apply,
)
from hololang.tensor.matrix import MatrixEngine, ParameterMultiplier
from hololang.tensor.transforms import Transform, TransformChain, ParallelTransformGroup
from hololang.tensor.hyperparams import HyperParameter, HyperParamSpace
from hololang.tensor.batch import SpreadTensor, BatchContainer

__all__ = [
    # helpers
    "strides", "flat_index", "total_elements",
    # core
    "Tensor", "SafeTensor",
    # graph
    "GraphNode", "ComputationGraph",
    # pool
    "TensorPool",
    # ops
    "relu", "leaky_relu", "sigmoid", "tanh_act", "elu", "softmax",
    "normalize", "layer_norm", "batch_norm", "min_max_scale",
    "dropout", "clip", "scale", "offset", "apply",
    # matrix
    "MatrixEngine", "ParameterMultiplier",
    # transforms
    "Transform", "TransformChain", "ParallelTransformGroup",
    # hyperparams
    "HyperParameter", "HyperParamSpace",
    # batch / shard
    "SpreadTensor", "BatchContainer",
]
