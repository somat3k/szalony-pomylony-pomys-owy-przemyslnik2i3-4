"""HoloLang AST node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class Node:
    """Base class for all AST nodes."""
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

@dataclass
class IntLiteral(Node):
    value: int

@dataclass
class FloatLiteral(Node):
    value: float

@dataclass
class StringLiteral(Node):
    value: str

@dataclass
class BoolLiteral(Node):
    value: bool

@dataclass
class NullLiteral(Node):
    pass

@dataclass
class ListLiteral(Node):
    elements: list[Node]

@dataclass
class DictLiteral(Node):
    pairs: list[tuple[Node, Node]]


# ---------------------------------------------------------------------------
# Identifiers & member access
# ---------------------------------------------------------------------------

@dataclass
class Identifier(Node):
    name: str

@dataclass
class MemberAccess(Node):
    """obj.member"""
    obj: Node
    member: str

@dataclass
class IndexAccess(Node):
    """obj[index]"""
    obj: Node
    index: Node

@dataclass
class Annotation(Node):
    """@name(args)"""
    name: str
    args: list[Node]
    kwargs: dict[str, Node]


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node

@dataclass
class UnaryOp(Node):
    op: str
    operand: Node

@dataclass
class CallExpr(Node):
    callee: Node
    args: list[Node]
    kwargs: dict[str, Node]

@dataclass
class PipeExpr(Node):
    """left -> right  or  left => right"""
    op: str          # "->" or "=>"
    left: Node
    right: Node

@dataclass
class AssignExpr(Node):
    op: str          # "=" or ":="
    target: Node
    value: Node

@dataclass
class RangeExpr(Node):
    """start..end"""
    start: Node
    end: Node

@dataclass
class TensorDimExpr(Node):
    """tensor Name[d0][d1]..."""
    name: str
    dims: list[Node]
    body: BlockStmt

@dataclass
class SpectrumExpr(Node):
    """spectrum(lo, hi)nm  or  visible-range"""
    lo: Node
    hi: Node
    unit: str = "nm"

@dataclass
class NmLiteral(Node):
    """450nm – a wavelength literal."""
    value: float


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class BlockStmt(Node):
    stmts: list[Node]

@dataclass
class LetStmt(Node):
    name: str
    is_const: bool
    value: Node | None

@dataclass
class ReturnStmt(Node):
    value: Node | None

@dataclass
class IfStmt(Node):
    condition: Node
    then_block: BlockStmt
    else_block: BlockStmt | None

@dataclass
class WhileStmt(Node):
    condition: Node
    body: BlockStmt

@dataclass
class ForStmt(Node):
    var: str
    iterable: Node
    body: BlockStmt

@dataclass
class BreakStmt(Node):
    pass

@dataclass
class ContinueStmt(Node):
    pass

@dataclass
class ExprStmt(Node):
    expr: Node

@dataclass
class ImportStmt(Node):
    path: str
    alias: str | None


# ---------------------------------------------------------------------------
# Top-level declarations
# ---------------------------------------------------------------------------

@dataclass
class FunctionDecl(Node):
    name: str
    params: list[tuple[str, str | None]]   # (name, type_hint)
    body: BlockStmt
    annotations: list[Annotation]

@dataclass
class DeviceDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt          # key: value pairs + nested blocks

@dataclass
class TensorDecl(Node):
    name: str
    dims: list[Node]
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class MeshDecl(Node):
    name: str
    source: Node | None      # optional backing tensor / device
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class CanvasDecl(Node):
    name: str
    source: Node | None
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class KernelDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class PoolDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class RuntimeDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class SessionDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class SkillDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class DocDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class EnumDecl(Node):
    name: str
    variants: list[tuple[str, Node | None]]  # (name, value_expr | None)

@dataclass
class ChannelDecl(Node):
    name: str
    protocol: str           # "grpc" | "ws" | "http"
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class WebhookDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class ApiDecl(Node):
    name: str
    annotations: list[Annotation]
    body: BlockStmt

@dataclass
class TileStmt(Node):
    row: Node
    grid_col: Node
    target: Node            # pipe target expression

@dataclass
class InvokeStmt(Node):
    target: Node
    pipe: Node | None       # optional  -> Canvas -> Device

@dataclass
class TransformStmt(Node):
    name: str
    args: list[Node]
    kwargs: dict[str, Node]

@dataclass
class BeamStmt(Node):
    args: list[Node]
    kwargs: dict[str, Node]

@dataclass
class ImpulseStmt(Node):
    from_tile: Node
    to_tile: Node
    payload: Node | None

@dataclass
class EmitStmt(Node):
    channel: Node
    message: Node

@dataclass
class ListenStmt(Node):
    channel: Node
    handler: Node

@dataclass
class ConnectStmt(Node):
    target: Node
    port: Node | None

@dataclass
class BindStmt(Node):
    name: str
    target: Node

@dataclass
class DebugStmt(Node):
    expr: Node

@dataclass
class ConfigStmt(Node):
    pairs: dict[str, Node]

@dataclass
class ParamStmt(Node):
    name: str
    value: Node

@dataclass
class Program(Node):
    """Root node – list of top-level declarations / statements."""
    body: list[Node]
