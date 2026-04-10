"""HoloLang tree-walking interpreter.

Evaluates a :class:`~hololang.lang.ast_nodes.Program` AST produced by
:mod:`hololang.lang.parser` inside a given :class:`Environment`.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable

from hololang.lang.ast_nodes import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Signal objects (control flow)
# ---------------------------------------------------------------------------

class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value

class _BreakSignal(Exception):
    pass

class _ContinueSignal(Exception):
    pass


# ---------------------------------------------------------------------------
# HoloLang runtime values
# ---------------------------------------------------------------------------

class HoloEnum:
    """Runtime representation of an enum declaration."""

    def __init__(self, name: str, variants: dict[str, Any]) -> None:
        self.name = name
        self.variants = variants

    def __repr__(self) -> str:
        return f"<enum {self.name}>"

    def __getattr__(self, item: str) -> Any:
        if item in ("name", "variants"):
            raise AttributeError(item)
        try:
            return self.variants[item]
        except KeyError:
            raise AttributeError(f"Enum {self.name!r} has no variant {item!r}")


class HoloObject:
    """Generic runtime object with named attributes."""

    def __init__(self, kind: str, name: str, attrs: dict[str, Any] | None = None) -> None:
        self.kind = kind
        self.name = name
        self.attrs: dict[str, Any] = attrs or {}

    def __repr__(self) -> str:
        return f"<{self.kind} {self.name!r}>"

    def get(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class HoloFunction:
    """User-defined function."""

    def __init__(self, name: str, params: list[tuple[str, str | None]],
                 body: BlockStmt, closure: "Environment") -> None:
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

    def __repr__(self) -> str:
        return f"<fn {self.name}>"


class HoloRange:
    """Integer or float range (start..end)."""

    def __init__(self, start: Any, end: Any) -> None:
        self.start = start
        self.end = end

    def __iter__(self):
        if isinstance(self.start, int) and isinstance(self.end, int):
            yield from range(self.start, self.end)
        else:
            # Yield floats in small steps
            steps = 100
            step = (self.end - self.start) / steps
            v = self.start
            for _ in range(steps):
                yield v
                v += step

    def __repr__(self) -> str:
        return f"<range {self.start}..{self.end}>"


# ---------------------------------------------------------------------------
# Environment (scope / symbol table)
# ---------------------------------------------------------------------------

class Environment:
    """Lexical scope chain."""

    def __init__(self, parent: "Environment | None" = None) -> None:
        self._vars: dict[str, Any] = {}
        self._parent = parent

    def define(self, name: str, value: Any) -> None:
        self._vars[name] = value

    def assign(self, name: str, value: Any) -> None:
        if name in self._vars:
            self._vars[name] = value
        elif self._parent is not None and name in self._parent:
            self._parent.assign(name, value)
        else:
            # Define in the current scope when no existing binding is found.
            self._vars[name] = value

    def lookup(self, name: str) -> Any:
        if name in self._vars:
            return self._vars[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise NameError(f"Undefined identifier {name!r}")

    def __contains__(self, name: str) -> bool:
        if name in self._vars:
            return True
        return name in self._parent if self._parent else False


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Evaluate a HoloLang AST.

    Parameters
    ----------
    env:
        Root :class:`Environment`.  If ``None`` a fresh one is created and
        populated with built-in functions.
    output_hook:
        Callable invoked whenever the interpreter produces output (debug,
        print, etc.).  Defaults to :func:`print`.
    """

    def __init__(
        self,
        env: Environment | None = None,
        output_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.env = env or Environment()
        self.output = output_hook or print
        self._setup_builtins()

    # ------------------------------------------------------------------
    # Built-in functions
    # ------------------------------------------------------------------

    def _setup_builtins(self) -> None:
        builtins: dict[str, Any] = {
            # I/O
            "print":   lambda *a: self.output(" ".join(str(x) for x in a)),
            "println": lambda *a: self.output(" ".join(str(x) for x in a)),
            # Maths
            "sqrt":  math.sqrt,
            "abs":   abs,
            "floor": math.floor,
            "ceil":  math.ceil,
            "round": round,
            "sin":   math.sin,
            "cos":   math.cos,
            "tan":   math.tan,
            "pi":    math.pi,
            "e":     math.e,
            # Type conversions
            "int":   int,
            "float": float,
            "str":   str,
            "bool":  bool,
            "list":  list,
            # Collections
            "len":   len,
            "range": lambda a, b=None: HoloRange(0 if b is None else a,
                                                 a if b is None else b),
            "append": lambda lst, v: lst.append(v) or lst,
            # Tensor helpers
            "zeros":    lambda *dims: self._make_tensor("zeros", list(dims)),
            "ones":     lambda *dims: self._make_tensor("ones", list(dims)),
            "fill":     lambda v, *dims: self._make_tensor("fill", list(dims), fill=v),
            # Mesh canvas helper – tile(row, col) returns (row, col) coordinates tuple
            "tile":     lambda row, col: (int(row), int(col)),
        }
        for name, val in builtins.items():
            self.env.define(name, val)

    def _make_tensor(self, mode: str, dims: list[int], fill: float = 0.0) -> list:
        """Create a nested list (stand-in tensor) without numpy."""
        if not dims:
            return fill if mode != "ones" else 1.0
        inner = [self._make_tensor(mode, dims[1:], fill) for _ in range(dims[0])]
        if mode == "ones" and len(dims) == 1:
            return [1.0] * dims[0]
        return inner

    # ------------------------------------------------------------------
    # Main eval entry points
    # ------------------------------------------------------------------

    def eval_program(self, program: Program) -> None:
        for node in program.body:
            self._eval(node, self.env)

    def eval_string(self, source: str) -> Any:
        from hololang.lang.parser import parse
        prog = parse(source)
        result: Any = None
        for node in prog.body:
            result = self._eval(node, self.env)
        return result

    # ------------------------------------------------------------------
    # Node dispatch
    # ------------------------------------------------------------------

    def _eval(self, node: Node, env: Environment) -> Any:
        method = "_eval_" + type(node).__name__
        handler = getattr(self, method, None)
        if handler is None:
            raise NotImplementedError(
                f"Interpreter: no handler for {type(node).__name__}"
            )
        return handler(node, env)

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def _eval_IntLiteral(self, node: IntLiteral, _env: Environment) -> int:
        return node.value

    def _eval_FloatLiteral(self, node: FloatLiteral, _env: Environment) -> float:
        return node.value

    def _eval_StringLiteral(self, node: StringLiteral, _env: Environment) -> str:
        return node.value

    def _eval_BoolLiteral(self, node: BoolLiteral, _env: Environment) -> bool:
        return node.value

    def _eval_NullLiteral(self, _node: NullLiteral, _env: Environment) -> None:
        return None

    def _eval_NmLiteral(self, node: NmLiteral, _env: Environment) -> float:
        return node.value

    def _eval_ListLiteral(self, node: ListLiteral, env: Environment) -> list:
        return [self._eval(e, env) for e in node.elements]

    def _eval_DictLiteral(self, node: DictLiteral, env: Environment) -> dict:
        return {self._eval(k, env): self._eval(v, env) for k, v in node.pairs}

    # ------------------------------------------------------------------
    # Identifiers & access
    # ------------------------------------------------------------------

    def _eval_Identifier(self, node: Identifier, env: Environment) -> Any:
        return env.lookup(node.name)

    def _eval_MemberAccess(self, node: MemberAccess, env: Environment) -> Any:
        obj = self._eval(node.obj, env)
        if isinstance(obj, HoloObject):
            return obj.attrs.get(node.member)
        if isinstance(obj, HoloEnum):
            return obj.variants.get(node.member)
        if isinstance(obj, dict):
            return obj.get(node.member)
        return getattr(obj, node.member, None)

    def _eval_IndexAccess(self, node: IndexAccess, env: Environment) -> Any:
        obj = self._eval(node.obj, env)
        idx = self._eval(node.index, env)
        return obj[idx]

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    _BINOPS: dict[str, Callable[[Any, Any], Any]] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "%": operator.mod,
        "==": operator.eq,
        "!=": operator.ne,
        "<":  operator.lt,
        ">":  operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "&":  operator.and_,
        "|":  operator.or_,
        "^":  operator.xor,
    }

    def _eval_BinaryOp(self, node: BinaryOp, env: Environment) -> Any:
        left  = self._eval(node.left, env)
        right = self._eval(node.right, env)
        op    = self._BINOPS.get(node.op)
        if op is None:
            raise ValueError(f"Unknown binary operator {node.op!r}")
        return op(left, right)

    def _eval_UnaryOp(self, node: UnaryOp, env: Environment) -> Any:
        val = self._eval(node.operand, env)
        if node.op == "-":
            return -val
        if node.op == "!":
            return not val
        if node.op == "~":
            return ~val
        raise ValueError(f"Unknown unary operator {node.op!r}")

    def _eval_RangeExpr(self, node: RangeExpr, env: Environment) -> HoloRange:
        start = self._eval(node.start, env)
        end   = self._eval(node.end, env)
        return HoloRange(start, end)

    def _eval_SpectrumExpr(self, node: SpectrumExpr, env: Environment) -> dict:
        lo = self._eval(node.lo, env)
        hi = self._eval(node.hi, env)
        return {"lo": lo, "hi": hi, "unit": node.unit}

    # ------------------------------------------------------------------
    # Calls & pipes
    # ------------------------------------------------------------------

    def _eval_CallExpr(self, node: CallExpr, env: Environment) -> Any:
        callee = self._eval(node.callee, env)
        args   = [self._eval(a, env) for a in node.args]
        kwargs = {k: self._eval(v, env) for k, v in node.kwargs.items()}
        if isinstance(callee, HoloFunction):
            return self._call_holo(callee, args, kwargs)
        if callable(callee):
            return callee(*args, **kwargs)
        raise TypeError(f"Cannot call {callee!r}")

    def _call_holo(self, fn: HoloFunction, args: list[Any],
                   kwargs: dict[str, Any]) -> Any:
        local = Environment(parent=fn.closure)
        for (p_name, _), val in zip(fn.params, args):
            local.define(p_name, val)
        for k, v in kwargs.items():
            local.define(k, v)
        try:
            self._eval_BlockStmt(fn.body, local)
        except _ReturnSignal as ret:
            return ret.value
        return None

    def _eval_PipeExpr(self, node: PipeExpr, env: Environment) -> Any:
        """Evaluate a pipe chain: left -> right.

        The right-hand side receives the left-hand value as its first
        argument when it resolves to a callable, or simply becomes the
        output if it's a device / object.
        """
        left = self._eval(node.left, env)
        right_node = node.right

        # If right is an identifier or member access, try calling it with left
        if isinstance(right_node, (Identifier, MemberAccess)):
            right = self._eval(right_node, env)
            if isinstance(right, HoloFunction):
                return self._call_holo(right, [left], {})
            if callable(right):
                return right(left)
            if isinstance(right, HoloObject):
                right.set("_input", left)
                return right
            return right

        if isinstance(right_node, CallExpr):
            callee = self._eval(right_node.callee, env)
            args   = [left] + [self._eval(a, env) for a in right_node.args]
            kwargs = {k: self._eval(v, env) for k, v in right_node.kwargs.items()}
            if callable(callee):
                return callee(*args, **kwargs)
            return callee

        right = self._eval(right_node, env)
        return right

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _eval_AssignExpr(self, node: AssignExpr, env: Environment) -> Any:
        value = self._eval(node.value, env)
        self._assign_target(node.target, value, env)
        return value

    def _assign_target(self, target: Node, value: Any, env: Environment) -> None:
        if isinstance(target, Identifier):
            env.assign(target.name, value)
        elif isinstance(target, MemberAccess):
            obj = self._eval(target.obj, env)
            if isinstance(obj, HoloObject):
                obj.set(target.member, value)
            elif isinstance(obj, dict):
                obj[target.member] = value
            else:
                setattr(obj, target.member, value)
        elif isinstance(target, IndexAccess):
            obj = self._eval(target.obj, env)
            idx = self._eval(target.index, env)
            obj[idx] = value
        else:
            raise ValueError(f"Invalid assignment target {type(target).__name__}")

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _eval_BlockStmt(self, node: BlockStmt, env: Environment) -> None:
        for stmt in node.stmts:
            self._eval(stmt, env)

    def _eval_LetStmt(self, node: LetStmt, env: Environment) -> None:
        value = self._eval(node.value, env) if node.value is not None else None
        env.define(node.name, value)

    def _eval_ReturnStmt(self, node: ReturnStmt, env: Environment) -> None:
        value = self._eval(node.value, env) if node.value is not None else None
        raise _ReturnSignal(value)

    def _eval_IfStmt(self, node: IfStmt, env: Environment) -> None:
        cond = self._eval(node.condition, env)
        if cond:
            local = Environment(parent=env)
            self._eval_BlockStmt(node.then_block, local)
        elif node.else_block is not None:
            local = Environment(parent=env)
            self._eval_BlockStmt(node.else_block, local)

    def _eval_WhileStmt(self, node: WhileStmt, env: Environment) -> None:
        while self._eval(node.condition, env):
            local = Environment(parent=env)
            try:
                self._eval_BlockStmt(node.body, local)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def _eval_ForStmt(self, node: ForStmt, env: Environment) -> None:
        iterable = self._eval(node.iterable, env)
        for item in iterable:
            local = Environment(parent=env)
            local.define(node.var, item)
            try:
                self._eval_BlockStmt(node.body, local)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def _eval_BreakStmt(self, _node: BreakStmt, _env: Environment) -> None:
        raise _BreakSignal()

    def _eval_ContinueStmt(self, _node: ContinueStmt, _env: Environment) -> None:
        raise _ContinueSignal()

    def _eval_ExprStmt(self, node: ExprStmt, env: Environment) -> Any:
        return self._eval(node.expr, env)

    def _eval_ImportStmt(self, node: ImportStmt, env: Environment) -> None:
        import os
        path = node.path
        alias = node.alias or path.split("/")[-1].replace(".hl", "")

        if path.endswith(".hl"):
            # Resolve relative paths against the current working directory
            resolved = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            if os.path.exists(resolved):
                from hololang.lang.parser import parse
                with open(resolved, encoding="utf-8") as fh:
                    source = fh.read()
                prog = parse(source)
                module_env = Environment()
                sub = Interpreter(env=module_env, output_hook=self.output)
                sub.eval_program(prog)
                module_obj = HoloObject("module", alias)
                module_obj.attrs.update(
                    {k: v for k, v in module_env._vars.items()}
                )
                env.define(alias, module_obj)
            else:
                # File not found – define as an empty module placeholder
                env.define(alias, HoloObject("module", alias))
        else:
            # Python module import
            import importlib
            try:
                mod = importlib.import_module(path.replace("/", ".").rstrip(".hl"))
                env.define(alias, mod)
            except ModuleNotFoundError:
                env.define(alias, HoloObject("module", alias))

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _eval_FunctionDecl(self, node: FunctionDecl, env: Environment) -> None:
        fn = HoloFunction(node.name, node.params, node.body, env)
        env.define(node.name, fn)

    def _eval_DeviceDecl(self, node: DeviceDecl, env: Environment) -> None:
        obj = HoloObject("device", node.name)
        local = Environment(parent=env)
        local.define("self", obj)
        self._eval_BlockStmt(node.body, local)
        # Hoist local vars into obj.attrs
        for k, v in local._vars.items():
            if k != "self":
                obj.set(k, v)
        env.define(node.name, obj)

    def _eval_TensorDecl(self, node: TensorDecl, env: Environment) -> None:
        dims = [self._eval(d, env) for d in node.dims]
        from hololang.tensor.tensor import Tensor
        t = Tensor(dims=dims, name=node.name)
        local = Environment(parent=env)
        local.define("self", t)
        self._eval_BlockStmt(node.body, local)
        for k, v in local._vars.items():
            if k != "self":
                t.meta[k] = v
        env.define(node.name, t)

    def _eval_MeshDecl(self, node: MeshDecl, env: Environment) -> None:
        from hololang.mesh.canvas import Canvas
        source = self._eval(node.source, env) if node.source else None
        canvas = Canvas(name=node.name, source=source)
        local = Environment(parent=env)
        local.define("self", canvas)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, canvas)

    def _eval_CanvasDecl(self, node: CanvasDecl, env: Environment) -> None:
        from hololang.mesh.canvas import Canvas
        source = self._eval(node.source, env) if node.source else None
        canvas = Canvas(name=node.name, source=source)
        local = Environment(parent=env)
        local.define("self", canvas)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, canvas)

    def _eval_KernelDecl(self, node: KernelDecl, env: Environment) -> None:
        from hololang.vm.kernel import Kernel
        k = Kernel(name=node.name)
        local = Environment(parent=env)
        local.define("self", k)
        self._eval_BlockStmt(node.body, local)
        for kk, v in local._vars.items():
            if kk != "self":
                k.params[kk] = v
        env.define(node.name, k)

    def _eval_PoolDecl(self, node: PoolDecl, env: Environment) -> None:
        from hololang.vm.runtime import PoolRuntime
        p = PoolRuntime(name=node.name)
        local = Environment(parent=env)
        local.define("self", p)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, p)

    def _eval_RuntimeDecl(self, node: RuntimeDecl, env: Environment) -> None:
        from hololang.vm.runtime import PoolRuntime
        r = PoolRuntime(name=node.name)
        local = Environment(parent=env)
        local.define("self", r)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, r)

    def _eval_SessionDecl(self, node: SessionDecl, env: Environment) -> None:
        from hololang.docs.session import Session
        sess = Session(name=node.name)
        local = Environment(parent=env)
        local.define("self", sess)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, sess)

    def _eval_SkillDecl(self, node: SkillDecl, env: Environment) -> None:
        obj = HoloObject("skill", node.name)
        local = Environment(parent=env)
        local.define("self", obj)
        self._eval_BlockStmt(node.body, local)
        for k, v in local._vars.items():
            if k != "self":
                obj.set(k, v)
        env.define(node.name, obj)

    def _eval_DocDecl(self, node: DocDecl, env: Environment) -> None:
        from hololang.docs.directory import DocDirectory
        d = DocDirectory(name=node.name)
        local = Environment(parent=env)
        local.define("self", d)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, d)

    def _eval_EnumDecl(self, node: EnumDecl, env: Environment) -> None:
        variants: dict[str, Any] = {}
        for i, (v_name, v_val) in enumerate(node.variants):
            if v_val is not None:
                variants[v_name] = self._eval(v_val, env)
            else:
                variants[v_name] = i
        env.define(node.name, HoloEnum(node.name, variants))

    def _eval_ChannelDecl(self, node: ChannelDecl, env: Environment) -> None:
        from hololang.network.api import Channel
        ch = Channel(name=node.name, protocol=node.protocol)
        local = Environment(parent=env)
        local.define("self", ch)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, ch)

    def _eval_WebhookDecl(self, node: WebhookDecl, env: Environment) -> None:
        from hololang.network.webhook import Webhook
        wh = Webhook(name=node.name)
        local = Environment(parent=env)
        local.define("self", wh)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, wh)

    def _eval_ApiDecl(self, node: ApiDecl, env: Environment) -> None:
        from hololang.network.api import ApiEndpoint
        api = ApiEndpoint(name=node.name)
        local = Environment(parent=env)
        local.define("self", api)
        self._eval_BlockStmt(node.body, local)
        env.define(node.name, api)

    # ------------------------------------------------------------------
    # Special statements
    # ------------------------------------------------------------------

    def _eval_TileStmt(self, node: TileStmt, env: Environment) -> None:
        from hololang.mesh.tile import Tile
        row    = self._eval(node.row, env)
        col    = self._eval(node.grid_col, env)
        target = self._eval(node.target, env)
        tile   = Tile(row=int(row), col=int(col), data=target)
        # Register with most-recent canvas in scope
        try:
            canvas = env.lookup("self")
            if hasattr(canvas, "add_tile"):
                canvas.add_tile(tile)
        except NameError:
            pass
        env.define(f"tile_{row}_{col}", tile)

    def _eval_InvokeStmt(self, node: InvokeStmt, env: Environment) -> None:
        if node.pipe is not None:
            result = self._eval(node.pipe, env)
        else:
            result = self._eval(node.target, env)
        self.output(f"[invoke] {result}")

    def _eval_TransformStmt(self, node: TransformStmt, env: Environment) -> None:
        args   = [self._eval(a, env) for a in node.args]
        kwargs = {k: self._eval(v, env) for k, v in node.kwargs.items()}
        from hololang.tensor.tensor import Tensor
        self.output(f"[transform:{node.name}] args={args} kwargs={kwargs}")

    def _eval_BeamStmt(self, node: BeamStmt, env: Environment) -> None:
        args   = [self._eval(a, env) for a in node.args]
        kwargs = {k: self._eval(v, env) for k, v in node.kwargs.items()}
        self.output(f"[beam] r={args[0] if args else '?'} "
                    f"g={args[1] if len(args) > 1 else '?'} "
                    f"b={args[2] if len(args) > 2 else '?'} "
                    f"kwargs={kwargs}")

    def _eval_ImpulseStmt(self, node: ImpulseStmt, env: Environment) -> None:
        from hololang.mesh.tile import Tile
        from_t  = self._eval(node.from_tile, env)
        to_t    = self._eval(node.to_tile, env)
        payload = self._eval(node.payload, env) if node.payload else None

        # Wire canvas tiles when both endpoints are (row, col) tuples
        if isinstance(from_t, tuple) and isinstance(to_t, tuple):
            fr, fc = int(from_t[0]), int(from_t[1])
            tr, tc = int(to_t[0]), int(to_t[1])

            src_tile: Tile | None = None
            try:
                candidate = env.lookup(f"tile_{fr}_{fc}")
                if isinstance(candidate, Tile):
                    src_tile = candidate
            except NameError:
                pass

            if src_tile is not None:
                src_tile.connect_to(tr, tc)
                # Deliver the initial payload to the destination tile
                if payload is not None:
                    try:
                        dst = env.lookup(f"tile_{tr}_{tc}")
                        if isinstance(dst, Tile):
                            dst.receive_impulse(payload)
                    except NameError:
                        pass

        self.output(f"[impulse] {from_t} -> {to_t} payload={payload}")

    def _eval_EmitStmt(self, node: EmitStmt, env: Environment) -> None:
        channel = self._eval(node.channel, env)
        message = self._eval(node.message, env)
        if hasattr(channel, "emit"):
            channel.emit(message)
        self.output(f"[emit] channel={channel} message={message}")

    def _eval_ListenStmt(self, node: ListenStmt, env: Environment) -> None:
        channel = self._eval(node.channel, env)
        handler = self._eval(node.handler, env)
        if hasattr(channel, "listen"):
            channel.listen(handler)
        self.output(f"[listen] channel={channel} handler={handler}")

    def _eval_ConnectStmt(self, node: ConnectStmt, env: Environment) -> None:
        target = self._eval(node.target, env)
        port   = self._eval(node.port, env) if node.port else None
        self.output(f"[connect] {target}:{port}")

    def _eval_BindStmt(self, node: BindStmt, env: Environment) -> None:
        target = self._eval(node.target, env)
        env.define(node.name, target)
        self.output(f"[bind] {node.name} = {target}")

    def _eval_DebugStmt(self, node: DebugStmt, env: Environment) -> None:
        val = self._eval(node.expr, env)
        self.output(f"[debug] {val!r}")

    def _eval_ConfigStmt(self, node: ConfigStmt, env: Environment) -> None:
        for k, v_node in node.pairs.items():
            v = self._eval(v_node, env)
            env.define(k, v)
            self.output(f"[config] {k} = {v!r}")

    def _eval_ParamStmt(self, node: ParamStmt, env: Environment) -> None:
        value = self._eval(node.value, env)
        env.define(node.name, value)
        self.output(f"[param] {node.name} = {value!r}")

    def _eval_Annotation(self, node: Annotation, env: Environment) -> None:
        # Annotations are metadata only; evaluated but not executed
        pass
