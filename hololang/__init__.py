"""
HoloLang – Holographic Device Control Language
================================================
A domain-specific language and runtime framework for:

* Holographic display devices driven by laser beams and galvanised mirrors
* Tensor / SafeTensor processing with computation graphs and pooled runtimes
* Replicable-kernel virtual machine with a block-controller generative engine
* Multi-solution MDI mesh-canvas with tile-to-tile impulse cycles
* Network layer: gRPC channels, WebSocket streams, and webhook dispatchers
* Session / skill / documental-directory management

Quick start::

    from hololang import HoloRuntime
    rt = HoloRuntime()
    rt.run_file("my_program.hl")
"""

from hololang.runtime import HoloRuntime

__all__ = ["HoloRuntime"]
__version__ = "0.1.0"
