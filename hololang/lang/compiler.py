"""HoloLang AST → Kernel bytecode compiler.

Translates a subset of the HoloLang AST into a flat sequence of
:class:`~hololang.vm.kernel.Instruction` objects that can be loaded
directly into a :class:`~hololang.vm.kernel.Kernel` and executed.

**Supported constructs**

* Literals: ``int``, ``float``, ``string``, ``bool``, ``null``, ``nm``
* Variables: ``let`` / ``const`` declarations and identifier load/store
* Binary operators: ``+ - * / % == != < > <= >=``
* Unary operators: ``-`` ``!``
* Assignments (simple name targets)
* ``if`` / ``else``
* ``while``
* ``for var in range(n)`` and ``for var in start..end``
* Function declarations (compiled to subroutines) and calls
* ``return``
* ``debug`` statement (logs top-of-stack)

**Calling convention**

Arguments are stored in registers ``__arg_0__``, ``__arg_1__``, … before
``CALL``.  The callee's prologue loads them into the param names.  The
return value is left on the operand stack after ``CALL`` returns.

**Unsupported constructs** (domain declarations such as ``device``,
``tensor``, ``mesh``, ``channel``, ``import``, etc.) are silently
skipped, which allows mixed programs to be partially compiled.

Usage::

    from hololang.lang.parser import parse
    from hololang.lang.compiler import Compiler
    from hololang.vm.kernel import Kernel

    prog = parse('''
        fn factorial(n) {
            if n <= 1 { return 1 }
            return n * factorial(n - 1)
        }
        let r = factorial(5)
        debug r
    ''')

    compiler = Compiler()
    instructions = compiler.compile(prog)

    k = Kernel("example")
    k.load(instructions)
    k.run()
    # k.get_log() contains "[example] 120"
"""

from __future__ import annotations

from hololang.lang.ast_nodes import (
    Node, Program,
    # literals
    IntLiteral, FloatLiteral, StringLiteral, BoolLiteral, NullLiteral,
    NmLiteral, ListLiteral, DictLiteral,
    # identifiers
    Identifier, MemberAccess, IndexAccess, Annotation,
    # operators
    BinaryOp, UnaryOp, RangeExpr, SpectrumExpr,
    # calls
    CallExpr, PipeExpr, AssignExpr,
    # statements
    BlockStmt, LetStmt, ReturnStmt, IfStmt, WhileStmt, ForStmt,
    BreakStmt, ContinueStmt, ExprStmt, DebugStmt,
    # declarations
    FunctionDecl,
    # domain declarations (skipped at compile time)
    DeviceDecl, TensorDecl, MeshDecl, CanvasDecl, KernelDecl,
    PoolDecl, RuntimeDecl, SessionDecl, SkillDecl, DocDecl,
    ChannelDecl, WebhookDecl, ApiDecl, EnumDecl, ImportStmt,
)
from hololang.vm.kernel import Instruction


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CompileError(Exception):
    """Raised when compilation fails."""


# ---------------------------------------------------------------------------
# Break / continue sentinel (used to patch jump targets)
# ---------------------------------------------------------------------------

class _LoopContext:
    """Tracks pending JMP/JZ patch sites for a loop body."""

    def __init__(self) -> None:
        self.break_sites:    list[int] = []   # indices of JMP(0) for break
        self.continue_sites: list[int] = []   # indices of JMP(0) for continue


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

_DOMAIN_DECLS = (
    DeviceDecl, TensorDecl, MeshDecl, CanvasDecl, KernelDecl,
    PoolDecl, RuntimeDecl, SessionDecl, SkillDecl, DocDecl,
    ChannelDecl, WebhookDecl, ApiDecl, EnumDecl, ImportStmt, Annotation,
)

_BINOP_OPCODES: dict[str, str] = {
    "+":  "ADD",
    "-":  "SUB",
    "*":  "MUL",
    "/":  "DIV",
    "%":  "MOD",
    "==": "EQ",
    "<":  "LT",
    ">":  "GT",
}


class Compiler:
    """Compile a HoloLang :class:`~hololang.lang.ast_nodes.Program` to a
    list of :class:`~hololang.vm.kernel.Instruction` objects.
    """

    def __init__(self) -> None:
        self._code:         list[Instruction] = []
        self._fn_addrs:     dict[str, int]    = {}   # fn_name -> start index
        self._call_patches: list[tuple[int, str]] = []  # (instr_idx, fn_name)
        self._loop_stack:   list[_LoopContext] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compile(self, program: Program) -> list[Instruction]:
        """Compile *program* and return the instruction list.

        All function declarations are placed before the main body (each
        preceded by a skip-JMP so they are not executed during init).
        """
        self._code          = []
        self._fn_addrs      = {}
        self._call_patches  = []
        self._loop_stack    = []

        fn_decls = [n for n in program.body if isinstance(n, FunctionDecl)]
        others   = [n for n in program.body if not isinstance(n, FunctionDecl)]

        # Emit a JMP that skips over all function bodies so they are only
        # executed when explicitly CALLed.
        jmp_to_body = self._emit("JMP", 0)  # placeholder; patched below

        for fn in fn_decls:
            self._compile_fn_body(fn)

        # Patch the initial JMP to the start of the main body
        self._patch(jmp_to_body, len(self._code))

        for node in others:
            self._compile_node(node)

        self._emit("HALT")

        # Back-patch call sites
        for idx, fn_name in self._call_patches:
            if fn_name not in self._fn_addrs:
                raise CompileError(f"Undefined function {fn_name!r}")
            self._patch(idx, self._fn_addrs[fn_name])

        return list(self._code)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, opcode: str, *operands) -> int:
        """Append an instruction; return its index in the code list."""
        idx = len(self._code)
        self._code.append(Instruction(opcode, operands if operands else ()))
        return idx

    def _patch(self, idx: int, target: int) -> None:
        """Replace the first operand of instruction *idx* with *target*."""
        old = self._code[idx]
        self._code[idx] = Instruction(old.opcode, (target,))

    # ------------------------------------------------------------------
    # Statement compilation
    # ------------------------------------------------------------------

    def _compile_node(self, node: Node) -> None:
        """Dispatch to a statement or domain-decl compiler."""
        if isinstance(node, _DOMAIN_DECLS):
            return  # silently skip domain declarations
        method = f"_compile_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler is None:
            raise CompileError(
                f"Compiler: no handler for {type(node).__name__}"
            )
        handler(node)

    def _compile_BlockStmt(self, node: BlockStmt) -> None:
        for stmt in node.stmts:
            self._compile_node(stmt)

    def _compile_LetStmt(self, node: LetStmt) -> None:
        if node.value is not None:
            self._compile_expr(node.value)
        else:
            self._emit("PUSH", None)
        self._emit("STORE", node.name)

    def _compile_ReturnStmt(self, node: ReturnStmt) -> None:
        if node.value is not None:
            self._compile_expr(node.value)
        else:
            self._emit("PUSH", None)
        self._emit("RET")

    def _compile_IfStmt(self, node: IfStmt) -> None:
        self._compile_expr(node.condition)
        jz_idx = self._emit("JZ", 0)           # placeholder

        self._compile_BlockStmt(node.then_block)

        if node.else_block is not None:
            jmp_idx = self._emit("JMP", 0)     # skip else
            self._patch(jz_idx, len(self._code))
            self._compile_BlockStmt(node.else_block)
            self._patch(jmp_idx, len(self._code))
        else:
            self._patch(jz_idx, len(self._code))

    def _compile_WhileStmt(self, node: WhileStmt) -> None:
        ctx = _LoopContext()
        self._loop_stack.append(ctx)

        loop_start = len(self._code)
        self._compile_expr(node.condition)
        jz_idx = self._emit("JZ", 0)           # exit placeholder

        self._compile_BlockStmt(node.body)

        # patch continue sites to here (re-evaluate condition)
        cont_target = len(self._code)
        for site in ctx.continue_sites:
            self._patch(site, cont_target)

        self._emit("JMP", loop_start)

        loop_end = len(self._code)
        self._patch(jz_idx, loop_end)
        for site in ctx.break_sites:
            self._patch(site, loop_end)

        self._loop_stack.pop()

    def _compile_ForStmt(self, node: ForStmt) -> None:
        """Compile ``for var in range(n)`` or ``for var in start..end``."""
        iterable = node.iterable
        start_reg = f"__for_start_{node.var}__"
        end_reg   = f"__for_end_{node.var}__"

        if isinstance(iterable, RangeExpr):
            self._compile_expr(iterable.start)
            self._emit("STORE", start_reg)
            self._compile_expr(iterable.end)
            self._emit("STORE", end_reg)
        elif (
            isinstance(iterable, CallExpr)
            and isinstance(iterable.callee, Identifier)
            and iterable.callee.name == "range"
        ):
            args = iterable.args
            if len(args) == 1:
                self._emit("PUSH", 0)
                self._emit("STORE", start_reg)
                self._compile_expr(args[0])
                self._emit("STORE", end_reg)
            elif len(args) == 2:
                self._compile_expr(args[0])
                self._emit("STORE", start_reg)
                self._compile_expr(args[1])
                self._emit("STORE", end_reg)
            else:
                raise CompileError("range() takes 1 or 2 arguments in for loops")
        else:
            raise CompileError(
                f"ForStmt: compiler supports only range() and a..b iterables; "
                f"got {type(iterable).__name__}"
            )

        # initialise loop variable
        self._emit("LOAD", start_reg)
        self._emit("STORE", node.var)

        ctx = _LoopContext()
        self._loop_stack.append(ctx)

        loop_start = len(self._code)

        # condition: var < end
        self._emit("LOAD", node.var)
        self._emit("LOAD", end_reg)
        self._emit("LT")
        jz_idx = self._emit("JZ", 0)           # exit placeholder

        self._compile_BlockStmt(node.body)

        # continue target: increment and loop
        cont_target = len(self._code)
        for site in ctx.continue_sites:
            self._patch(site, cont_target)

        # increment
        self._emit("LOAD", node.var)
        self._emit("PUSH", 1)
        self._emit("ADD")
        self._emit("STORE", node.var)
        self._emit("JMP", loop_start)

        loop_end = len(self._code)
        self._patch(jz_idx, loop_end)
        for site in ctx.break_sites:
            self._patch(site, loop_end)

        self._loop_stack.pop()

    def _compile_BreakStmt(self, _node: BreakStmt) -> None:
        if not self._loop_stack:
            raise CompileError("break outside loop")
        site = self._emit("JMP", 0)
        self._loop_stack[-1].break_sites.append(site)

    def _compile_ContinueStmt(self, _node: ContinueStmt) -> None:
        if not self._loop_stack:
            raise CompileError("continue outside loop")
        site = self._emit("JMP", 0)
        self._loop_stack[-1].continue_sites.append(site)

    def _compile_ExprStmt(self, node: ExprStmt) -> None:
        expr = node.expr
        if isinstance(expr, AssignExpr) and isinstance(expr.target, Identifier):
            # Statement-level assignment: store and discard
            self._compile_expr(expr.value)
            self._emit("STORE", expr.target.name)
        else:
            self._compile_expr(expr)
            self._emit("POP")   # discard the expression value

    def _compile_DebugStmt(self, node: DebugStmt) -> None:
        self._compile_expr(node.expr)
        self._emit("LOG")   # LOG peeks (no pop)
        self._emit("POP")   # discard

    def _compile_FunctionDecl(self, _node: FunctionDecl) -> None:
        pass  # function bodies are pre-compiled by compile()

    # ------------------------------------------------------------------
    # Function body compilation (called once per fn decl)
    # ------------------------------------------------------------------

    def _compile_fn_body(self, node: FunctionDecl) -> None:
        fn_start = len(self._code)
        self._fn_addrs[node.name] = fn_start

        # Prologue: load __arg_N__ registers into parameter names
        for i, (p_name, _) in enumerate(node.params):
            self._emit("LOAD", f"__arg_{i}__")
            self._emit("STORE", p_name)

        # Body
        for stmt in node.body.stmts:
            self._compile_node(stmt)

        # Implicit return None if execution falls off the end
        self._emit("PUSH", None)
        self._emit("RET")

    # ------------------------------------------------------------------
    # Expression compilation (leaves result on operand stack)
    # ------------------------------------------------------------------

    def _compile_expr(self, node: Node) -> None:
        method = f"_compile_expr_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler is None:
            raise CompileError(
                f"Compiler: cannot compile expression {type(node).__name__}"
            )
        handler(node)

    def _compile_expr_IntLiteral(self, n: IntLiteral) -> None:
        self._emit("PUSH", n.value)

    def _compile_expr_FloatLiteral(self, n: FloatLiteral) -> None:
        self._emit("PUSH", n.value)

    def _compile_expr_StringLiteral(self, n: StringLiteral) -> None:
        self._emit("PUSH", n.value)

    def _compile_expr_BoolLiteral(self, n: BoolLiteral) -> None:
        self._emit("PUSH", n.value)

    def _compile_expr_NullLiteral(self, _n: NullLiteral) -> None:
        self._emit("PUSH", None)

    def _compile_expr_NmLiteral(self, n: NmLiteral) -> None:
        self._emit("PUSH", n.value)

    def _compile_expr_Identifier(self, n: Identifier) -> None:
        self._emit("LOAD", n.name)

    def _compile_expr_BinaryOp(self, n: BinaryOp) -> None:
        self._compile_expr(n.left)
        self._compile_expr(n.right)

        # Handle operators not natively in the opcode table
        if n.op == "!=":
            self._emit("EQ")
            self._emit("NOT")
            return
        if n.op == "<=":
            # a <= b  ⟺  not (a > b)
            self._emit("GT")
            self._emit("NOT")
            return
        if n.op == ">=":
            # a >= b  ⟺  not (a < b)
            self._emit("LT")
            self._emit("NOT")
            return
        if n.op in ("&&", "and"):
            self._emit("AND")
            return
        if n.op in ("||", "or"):
            self._emit("OR")
            return

        opcode = _BINOP_OPCODES.get(n.op)
        if opcode is None:
            raise CompileError(f"Binary op {n.op!r} not supported by compiler")
        self._emit(opcode)

    def _compile_expr_UnaryOp(self, n: UnaryOp) -> None:
        self._compile_expr(n.operand)
        if n.op == "-":
            self._emit("NEG")
        elif n.op == "!":
            self._emit("NOT")
        else:
            raise CompileError(f"Unary op {n.op!r} not supported by compiler")

    def _compile_expr_AssignExpr(self, n: AssignExpr) -> None:
        """Assignment expression — leaves the assigned value on the stack."""
        self._compile_expr(n.value)
        if not isinstance(n.target, Identifier):
            raise CompileError(
                "Compiler only supports simple variable assignment targets"
            )
        self._emit("DUP")           # keep a copy as the expression's value
        self._emit("STORE", n.target.name)

    def _compile_expr_CallExpr(self, n: CallExpr) -> None:
        if not isinstance(n.callee, Identifier):
            raise CompileError(
                "Compiler only supports simple function calls by name"
            )
        fn_name = n.callee.name

        # Built-in: print / println — log the concatenated args
        if fn_name in ("print", "println"):
            if n.args:
                # Compile all args and concatenate as strings
                self._compile_expr(n.args[0])
                for arg in n.args[1:]:
                    self._compile_expr(arg)
                    self._emit("ADD")   # str + str concatenation
            else:
                self._emit("PUSH", "")
            self._emit("LOG")           # peek & log
            self._emit("POP")           # discard
            self._emit("PUSH", None)    # print() returns None
            return

        # Store each argument into __arg_N__ registers
        for i, arg in enumerate(n.args):
            self._compile_expr(arg)
            self._emit("STORE", f"__arg_{i}__")

        # Emit CALL with placeholder address; back-patch after compilation
        call_idx = self._emit("CALL", 0)
        self._call_patches.append((call_idx, fn_name))

    def _compile_expr_RangeExpr(self, _n: RangeExpr) -> None:
        raise CompileError(
            "RangeExpr cannot be compiled as a standalone value; "
            "use it directly in a for loop"
        )

    def _compile_expr_PipeExpr(self, n: PipeExpr) -> None:
        """Compile ``left -> right`` as a single-arg call when possible."""
        self._compile_expr(n.left)
        right = n.right
        if isinstance(right, Identifier):
            # Treat as single-argument call
            self._emit("STORE", "__arg_0__")
            call_idx = self._emit("CALL", 0)
            self._call_patches.append((call_idx, right.name))
        elif isinstance(right, CallExpr) and isinstance(right.callee, Identifier):
            self._emit("STORE", "__arg_0__")
            for i, arg in enumerate(right.args, start=1):
                self._compile_expr(arg)
                self._emit("STORE", f"__arg_{i}__")
            call_idx = self._emit("CALL", 0)
            self._call_patches.append((call_idx, right.callee.name))
        else:
            raise CompileError(
                f"PipeExpr: right-hand side {type(right).__name__} "
                "is not supported by the compiler"
            )
