"""HoloLang recursive-descent parser.

Converts a token stream (produced by :mod:`hololang.lang.lexer`) into an
AST rooted at :class:`~hololang.lang.ast_nodes.Program`.
"""

from __future__ import annotations

from typing import Callable

from hololang.lang.lexer import TT, Token, tokenize
from hololang.lang.ast_nodes import *  # noqa: F401,F403


class ParseError(Exception):
    """Raised when the parser encounters unexpected input."""


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        # Filter out bare newlines – they are only needed for error messages
        # at the lexer level; the grammar does not rely on them.
        self._tokens: list[Token] = [t for t in tokens if t.type != TT.NEWLINE]
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]  # EOF
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TT.EOF:
            self._pos += 1
        return tok

    def _check(self, *types: TT) -> bool:
        return self._peek().type in types

    def _match(self, *types: TT) -> Token | None:
        if self._check(*types):
            return self._advance()
        return None

    def _expect(self, tt: TT, msg: str = "") -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}{': ' + msg if msg else ''}, "
                f"got {tok.type.name} ({tok.value!r}) at line {tok.line}, col {tok.col}"
            )
        return self._advance()

    def _skip_semis(self) -> None:
        while self._match(TT.SEMICOLON):
            pass

    def _lc(self) -> tuple[int, int]:
        tok = self._peek()
        return tok.line, tok.col

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        line, col = self._lc()
        body: list[Node] = []
        self._skip_semis()
        while not self._check(TT.EOF):
            body.append(self._parse_top_level())
            self._skip_semis()
        return Program(body=body, line=line, col=col)

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _parse_top_level(self) -> Node:
        annotations: list[Annotation] = []
        while self._check(TT.AT):
            annotations.append(self._parse_annotation())

        tok = self._peek()
        tt = tok.type

        dispatch: dict[TT, Callable[..., Node]] = {
            TT.KW_DEVICE:   self._parse_device,
            TT.KW_TENSOR:   self._parse_tensor,
            TT.KW_MESH:     self._parse_mesh,
            TT.KW_CANVAS:   self._parse_canvas,
            TT.KW_KERNEL:   self._parse_kernel,
            TT.KW_POOL:     self._parse_pool,
            TT.KW_RUNTIME:  self._parse_runtime,
            TT.KW_SESSION:  self._parse_session,
            TT.KW_SKILL:    self._parse_skill,
            TT.KW_DOC:      self._parse_doc,
            TT.KW_ENUM:     self._parse_enum,
            TT.KW_CHANNEL:  self._parse_channel,
            TT.KW_WEBHOOK:  self._parse_webhook,
            TT.KW_API:      self._parse_api,
            TT.KW_FN:       self._parse_function,
            TT.KW_IMPORT:   self._parse_import,
        }
        if tt in dispatch:
            return dispatch[tt](annotations)
        if annotations:
            raise ParseError(
                f"Unexpected token {tt.name} after annotations at line {tok.line}"
            )
        return self._parse_statement()

    # ------------------------------------------------------------------
    # Annotation  @name(arg, key=val)
    # ------------------------------------------------------------------

    def _parse_annotation(self) -> Annotation:
        line, col = self._lc()
        self._expect(TT.AT)
        # Accept any keyword as an annotation name (e.g. @session, @kernel)
        tok = self._peek()
        if tok.type == TT.IDENT or tok.type.name.startswith("KW_"):
            self._advance()
            name = tok.value
        else:
            raise ParseError(
                f"Expected IDENT: annotation name, "
                f"got {tok.type.name} ({tok.value!r}) at line {tok.line}, col {tok.col}"
            )
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        if self._match(TT.LPAREN):
            while not self._check(TT.RPAREN, TT.EOF):
                if self._peek().type == TT.IDENT and self._peek(1).type == TT.EQ:
                    k = self._advance().value
                    self._advance()  # =
                    kwargs[k] = self._parse_expr()
                else:
                    args.append(self._parse_expr())
                if not self._match(TT.COMMA):
                    break
            self._expect(TT.RPAREN)
        return Annotation(name=name, args=args, kwargs=kwargs, line=line, col=col)

    # ------------------------------------------------------------------
    # Block  { stmt* }
    # ------------------------------------------------------------------

    def _parse_block(self) -> BlockStmt:
        line, col = self._lc()
        self._expect(TT.LBRACE)
        self._skip_semis()
        stmts: list[Node] = []
        while not self._check(TT.RBRACE, TT.EOF):
            stmts.append(self._parse_statement())
            self._skip_semis()
        self._expect(TT.RBRACE)
        return BlockStmt(stmts=stmts, line=line, col=col)

    # ------------------------------------------------------------------
    # Top-level declarations
    # ------------------------------------------------------------------

    def _named_decl(self) -> tuple[str, list[Node], int, int]:
        """Parse common 'keyword Name [dims] {...}' pattern."""
        line, col = self._lc()
        self._advance()  # consume keyword
        name = self._expect(TT.IDENT, "declaration name").value
        dims: list[Node] = []
        while self._match(TT.LBRACKET):
            dims.append(self._parse_expr())
            self._expect(TT.RBRACKET)
        return name, dims, line, col

    def _parse_device(self, annotations: list[Annotation]) -> DeviceDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return DeviceDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_tensor(self, annotations: list[Annotation]) -> TensorDecl:
        name, dims, line, col = self._named_decl()
        body = self._parse_block()
        return TensorDecl(name=name, dims=dims, annotations=annotations, body=body,
                          line=line, col=col)

    def _parse_mesh(self, annotations: list[Annotation]) -> MeshDecl:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        source: Node | None = None
        if self._match(TT.LPAREN):
            source = self._parse_expr()
            self._expect(TT.RPAREN)
        body = self._parse_block()
        return MeshDecl(name=name, source=source, annotations=annotations, body=body,
                        line=line, col=col)

    def _parse_canvas(self, annotations: list[Annotation]) -> CanvasDecl:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        source: Node | None = None
        if self._match(TT.LPAREN):
            source = self._parse_expr()
            self._expect(TT.RPAREN)
        body = self._parse_block()
        return CanvasDecl(name=name, source=source, annotations=annotations, body=body,
                          line=line, col=col)

    def _parse_kernel(self, annotations: list[Annotation]) -> KernelDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return KernelDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_pool(self, annotations: list[Annotation]) -> PoolDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return PoolDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_runtime(self, annotations: list[Annotation]) -> RuntimeDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return RuntimeDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_session(self, annotations: list[Annotation]) -> SessionDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return SessionDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_skill(self, annotations: list[Annotation]) -> SkillDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return SkillDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_doc(self, annotations: list[Annotation]) -> DocDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return DocDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_enum(self, _annotations: list[Annotation]) -> EnumDecl:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        variants: list[tuple[str, Node | None]] = []
        self._expect(TT.LBRACE)
        self._skip_semis()
        while not self._check(TT.RBRACE, TT.EOF):
            v_name = self._expect(TT.IDENT).value
            v_val: Node | None = None
            if self._match(TT.EQ):
                v_val = self._parse_expr()
            variants.append((v_name, v_val))
            if not self._match(TT.COMMA):
                self._skip_semis()
        self._expect(TT.RBRACE)
        return EnumDecl(name=name, variants=variants, line=line, col=col)

    def _parse_channel(self, annotations: list[Annotation]) -> ChannelDecl:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        protocol = "grpc"
        if self._match(TT.LPAREN):
            protocol = self._expect(TT.IDENT, "protocol name").value
            self._expect(TT.RPAREN)
        body = self._parse_block()
        return ChannelDecl(name=name, protocol=protocol, annotations=annotations,
                           body=body, line=line, col=col)

    def _parse_webhook(self, annotations: list[Annotation]) -> WebhookDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return WebhookDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_api(self, annotations: list[Annotation]) -> ApiDecl:
        name, _, line, col = self._named_decl()
        body = self._parse_block()
        return ApiDecl(name=name, annotations=annotations, body=body, line=line, col=col)

    def _parse_function(self, annotations: list[Annotation]) -> FunctionDecl:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        params: list[tuple[str, str | None]] = []
        self._expect(TT.LPAREN)
        while not self._check(TT.RPAREN, TT.EOF):
            p_name = self._expect(TT.IDENT).value
            p_type: str | None = None
            if self._match(TT.COLON):
                p_type = self._expect(TT.IDENT).value
            params.append((p_name, p_type))
            if not self._match(TT.COMMA):
                break
        self._expect(TT.RPAREN)
        body = self._parse_block()
        return FunctionDecl(name=name, params=params, body=body, annotations=annotations,
                            line=line, col=col)

    def _parse_import(self, _annotations: list[Annotation]) -> ImportStmt:
        line, col = self._lc()
        self._advance()
        path = self._expect(TT.STRING, "import path").value
        alias: str | None = None
        if self._peek().type == TT.IDENT and self._peek().value == "as":
            self._advance()
            alias = self._expect(TT.IDENT).value
        self._match(TT.SEMICOLON)
        return ImportStmt(path=path, alias=alias, line=line, col=col)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _parse_statement(self) -> Node:
        tok = self._peek()
        tt = tok.type

        stmt_map: dict[TT, Callable[[], Node]] = {
            TT.KW_LET:       self._parse_let,
            TT.KW_CONST:     self._parse_let,
            TT.KW_RETURN:    self._parse_return,
            TT.KW_IF:        self._parse_if,
            TT.KW_WHILE:     self._parse_while,
            TT.KW_FOR:       self._parse_for,
            TT.KW_BREAK:     lambda: (self._advance(), BreakStmt(line=tok.line, col=tok.col))[1],
            TT.KW_CONTINUE:  lambda: (self._advance(), ContinueStmt(line=tok.line, col=tok.col))[1],
            TT.KW_INVOKE:    self._parse_invoke,
            TT.KW_TRANSFORM: self._parse_transform,
            TT.KW_BEAM:      self._parse_beam,
            TT.KW_IMPULSE:   self._parse_impulse,
            TT.KW_EMIT:      self._parse_emit,
            TT.KW_LISTEN:    self._parse_listen,
            TT.KW_CONNECT:   self._parse_connect,
            TT.KW_BIND:      self._parse_bind,
            TT.KW_DEBUG:     self._parse_debug,
            TT.KW_CONFIG:    self._parse_config,
            TT.KW_PARAM:     self._parse_param,
            TT.KW_TILE:      self._parse_tile,
            TT.LBRACE:       self._parse_block,
        }
        if tt in stmt_map:
            return stmt_map[tt]()

        # Inline annotations inside blocks
        if tt == TT.AT:
            anns = []
            while self._check(TT.AT):
                anns.append(self._parse_annotation())
            inner = self._peek()
            if inner.type == TT.KW_FN:
                return self._parse_function(anns)
            # Otherwise treat as expression statement with annotations attached
            expr = self._parse_expr()
            self._match(TT.SEMICOLON)
            return ExprStmt(expr=expr, line=tok.line, col=tok.col)

        # key: value pairs inside config blocks (any IDENT or keyword followed by :)
        if (tt == TT.IDENT or tt.name.startswith("KW_")) and self._peek(1).type == TT.COLON:
            line, col = self._lc()
            key_tok = self._advance()
            self._advance()  # consume :
            value = self._parse_expr()
            self._match(TT.COMMA)
            self._match(TT.SEMICOLON)
            return ExprStmt(
                expr=AssignExpr(
                    op="=",
                    target=Identifier(name=key_tok.value, line=line, col=col),
                    value=value,
                    line=line,
                    col=col,
                ),
                line=line,
                col=col,
            )

        # Expression statement (assignment, call, pipe, …)
        expr = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ExprStmt(expr=expr, line=tok.line, col=tok.col)

    def _parse_let(self) -> LetStmt:
        line, col = self._lc()
        is_const = self._peek().type == TT.KW_CONST
        self._advance()
        name = self._expect(TT.IDENT).value
        value: Node | None = None
        if self._match(TT.EQ) or self._match(TT.COLON_EQ):
            value = self._parse_expr()
        self._match(TT.SEMICOLON)
        return LetStmt(name=name, is_const=is_const, value=value, line=line, col=col)

    def _parse_return(self) -> ReturnStmt:
        line, col = self._lc()
        self._advance()
        value: Node | None = None
        if not self._check(TT.SEMICOLON, TT.RBRACE, TT.EOF):
            value = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ReturnStmt(value=value, line=line, col=col)

    def _parse_if(self) -> IfStmt:
        line, col = self._lc()
        self._advance()
        cond = self._parse_expr()
        then = self._parse_block()
        else_block: BlockStmt | None = None
        if self._match(TT.KW_ELSE):
            if self._check(TT.KW_IF):
                else_block = BlockStmt(stmts=[self._parse_if()], line=line, col=col)
            else:
                else_block = self._parse_block()
        return IfStmt(condition=cond, then_block=then, else_block=else_block,
                      line=line, col=col)

    def _parse_while(self) -> WhileStmt:
        line, col = self._lc()
        self._advance()
        cond = self._parse_expr()
        body = self._parse_block()
        return WhileStmt(condition=cond, body=body, line=line, col=col)

    def _parse_for(self) -> ForStmt:
        line, col = self._lc()
        self._advance()
        var = self._expect(TT.IDENT).value
        self._expect(TT.KW_IN)
        iterable = self._parse_expr()
        body = self._parse_block()
        return ForStmt(var=var, iterable=iterable, body=body, line=line, col=col)

    def _parse_invoke(self) -> InvokeStmt:
        line, col = self._lc()
        self._advance()
        self._expect(TT.LPAREN)
        target = self._parse_expr()
        self._expect(TT.RPAREN)
        pipe: Node | None = None
        if self._check(TT.ARROW):
            # Consume full chain:  invoke(X) -> A -> B -> C
            pipe = target
            while self._check(TT.ARROW):
                pipe = self._parse_pipe_tail(pipe)
        self._match(TT.SEMICOLON)
        return InvokeStmt(target=target, pipe=pipe, line=line, col=col)

    def _parse_transform(self) -> TransformStmt:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT, "transform name").value
        # handle qualified form: transform.name(...)
        if self._match(TT.DOT):
            name = self._expect(TT.IDENT, "qualified transform name").value
        args, kwargs = self._parse_call_args()
        self._match(TT.SEMICOLON)
        return TransformStmt(name=name, args=args, kwargs=kwargs, line=line, col=col)

    def _parse_beam(self) -> BeamStmt:
        line, col = self._lc()
        self._advance()
        args, kwargs = self._parse_call_args()
        self._match(TT.SEMICOLON)
        return BeamStmt(args=args, kwargs=kwargs, line=line, col=col)

    def _parse_impulse(self) -> ImpulseStmt:
        line, col = self._lc()
        self._advance()
        self._expect(TT.LPAREN)
        from_t = self._parse_expr()
        self._expect(TT.COMMA)
        to_t = self._parse_expr()
        self._expect(TT.RPAREN)
        payload: Node | None = None
        if self._match(TT.ARROW):
            payload = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ImpulseStmt(from_tile=from_t, to_tile=to_t, payload=payload,
                           line=line, col=col)

    def _parse_emit(self) -> EmitStmt:
        line, col = self._lc()
        self._advance()
        # Parse channel as a postfix-level expression to avoid consuming ->
        channel = self._parse_postfix()
        self._expect(TT.ARROW)
        message = self._parse_expr()
        self._match(TT.SEMICOLON)
        return EmitStmt(channel=channel, message=message, line=line, col=col)

    def _parse_listen(self) -> ListenStmt:
        line, col = self._lc()
        self._advance()
        channel = self._parse_postfix()
        self._expect(TT.ARROW)
        handler = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ListenStmt(channel=channel, handler=handler, line=line, col=col)

    def _parse_connect(self) -> ConnectStmt:
        line, col = self._lc()
        self._advance()
        target = self._parse_expr()
        port: Node | None = None
        if self._match(TT.COLON):
            port = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ConnectStmt(target=target, port=port, line=line, col=col)

    def _parse_bind(self) -> BindStmt:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        self._expect(TT.EQ)
        target = self._parse_expr()
        self._match(TT.SEMICOLON)
        return BindStmt(name=name, target=target, line=line, col=col)

    def _parse_debug(self) -> DebugStmt:
        line, col = self._lc()
        self._advance()
        expr = self._parse_expr()
        self._match(TT.SEMICOLON)
        return DebugStmt(expr=expr, line=line, col=col)

    def _parse_config(self) -> ConfigStmt:
        line, col = self._lc()
        self._advance()
        self._expect(TT.LBRACE)
        pairs: dict[str, Node] = {}
        self._skip_semis()
        while not self._check(TT.RBRACE, TT.EOF):
            key = self._expect(TT.IDENT).value
            self._expect(TT.COLON)
            val = self._parse_expr()
            pairs[key] = val
            if not self._match(TT.COMMA):
                self._skip_semis()
        self._expect(TT.RBRACE)
        return ConfigStmt(pairs=pairs, line=line, col=col)

    def _parse_param(self) -> ParamStmt:
        line, col = self._lc()
        self._advance()
        name = self._expect(TT.IDENT).value
        self._expect(TT.COLON)
        value = self._parse_expr()
        self._match(TT.SEMICOLON)
        return ParamStmt(name=name, value=value, line=line, col=col)

    def _parse_tile(self) -> TileStmt:
        line, start_col = self._lc()
        self._advance()
        self._expect(TT.LPAREN)
        row = self._parse_expr()
        self._expect(TT.COMMA)
        col_expr = self._parse_expr()
        self._expect(TT.RPAREN)
        self._expect(TT.ARROW)
        target = self._parse_expr()
        self._match(TT.SEMICOLON)
        return TileStmt(row=row, grid_col=col_expr, target=target, line=line, col=start_col)

    # ------------------------------------------------------------------
    # Expression parsing  (Pratt / precedence-climbing)
    # ------------------------------------------------------------------

    def _parse_expr(self) -> Node:
        return self._parse_assignment()

    def _parse_assignment(self) -> Node:
        left = self._parse_pipe()
        if self._check(TT.EQ, TT.COLON_EQ):
            op_tok = self._advance()
            right = self._parse_assignment()
            return AssignExpr(op=op_tok.value, target=left, value=right,
                              line=op_tok.line, col=op_tok.col)
        return left

    def _parse_pipe(self) -> Node:
        left = self._parse_or()
        while self._check(TT.ARROW, TT.FAT_ARROW):
            left = self._parse_pipe_tail(left)
        return left

    def _parse_pipe_tail(self, left: Node) -> Node:
        op_tok = self._advance()
        right = self._parse_or()
        return PipeExpr(op=op_tok.value, left=left, right=right,
                        line=op_tok.line, col=op_tok.col)

    def _parse_or(self) -> Node:
        left = self._parse_and()
        while self._check(TT.PIPE):
            op = self._advance()
            right = self._parse_and()
            left = BinaryOp(op="|", left=left, right=right, line=op.line, col=op.col)
        return left

    def _parse_and(self) -> Node:
        left = self._parse_equality()
        while self._check(TT.AMPERSAND):
            op = self._advance()
            right = self._parse_equality()
            left = BinaryOp(op="&", left=left, right=right, line=op.line, col=op.col)
        return left

    def _parse_equality(self) -> Node:
        left = self._parse_comparison()
        while self._check(TT.EQEQ, TT.NEQ):
            op = self._advance()
            right = self._parse_comparison()
            left = BinaryOp(op=op.value, left=left, right=right,
                            line=op.line, col=op.col)
        return left

    def _parse_comparison(self) -> Node:
        left = self._parse_range()
        while self._check(TT.LT, TT.GT, TT.LTE, TT.GTE):
            op = self._advance()
            right = self._parse_range()
            left = BinaryOp(op=op.value, left=left, right=right,
                            line=op.line, col=op.col)
        return left

    def _parse_range(self) -> Node:
        left = self._parse_additive()
        if self._match(TT.DOTDOT):
            right = self._parse_additive()
            return RangeExpr(start=left, end=right, line=left.line, col=left.col)
        return left

    def _parse_additive(self) -> Node:
        left = self._parse_multiplicative()
        while self._check(TT.PLUS, TT.MINUS):
            op = self._advance()
            right = self._parse_multiplicative()
            left = BinaryOp(op=op.value, left=left, right=right,
                            line=op.line, col=op.col)
        return left

    def _parse_multiplicative(self) -> Node:
        left = self._parse_unary()
        while self._check(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self._advance()
            right = self._parse_unary()
            left = BinaryOp(op=op.value, left=left, right=right,
                            line=op.line, col=op.col)
        return left

    def _parse_unary(self) -> Node:
        if self._check(TT.BANG, TT.MINUS, TT.TILDE):
            op = self._advance()
            operand = self._parse_unary()
            return UnaryOp(op=op.value, operand=operand, line=op.line, col=op.col)
        return self._parse_postfix()

    def _parse_postfix(self) -> Node:
        expr = self._parse_primary()
        while True:
            if self._match(TT.DOT):
                # Allow any keyword or identifier after dot (member access)
                tok = self._peek()
                if tok.type == TT.IDENT or tok.type.name.startswith("KW_"):
                    self._advance()
                    member = tok.value
                else:
                    raise ParseError(
                        f"Expected IDENT: member name, "
                        f"got {tok.type.name} ({tok.value!r}) at line {tok.line}, col {tok.col}"
                    )
                expr = MemberAccess(obj=expr, member=member,
                                    line=expr.line, col=expr.col)
            elif self._check(TT.LPAREN):
                args, kwargs = self._parse_call_args()
                expr = CallExpr(callee=expr, args=args, kwargs=kwargs,
                                line=expr.line, col=expr.col)
            elif self._match(TT.LBRACKET):
                idx = self._parse_expr()
                self._expect(TT.RBRACKET)
                expr = IndexAccess(obj=expr, index=idx,
                                   line=expr.line, col=expr.col)
            else:
                break
        return expr

    def _parse_primary(self) -> Node:
        tok = self._peek()
        tt = tok.type

        if tt == TT.INTEGER:
            self._advance()
            # wavelength literal: 532nm
            if self._peek().type == TT.IDENT and self._peek().value in ("nm", "um", "mm"):
                unit = self._advance().value
                return NmLiteral(value=float(tok.value), line=tok.line, col=tok.col)
            return IntLiteral(value=tok.value, line=tok.line, col=tok.col)

        if tt == TT.FLOAT:
            self._advance()
            if self._peek().type == TT.IDENT and self._peek().value in ("nm", "um", "mm"):
                self._advance()
                return NmLiteral(value=tok.value, line=tok.line, col=tok.col)
            return FloatLiteral(value=tok.value, line=tok.line, col=tok.col)

        if tt == TT.STRING:
            self._advance()
            return StringLiteral(value=tok.value, line=tok.line, col=tok.col)

        if tt == TT.BOOL:
            self._advance()
            return BoolLiteral(value=tok.value, line=tok.line, col=tok.col)

        if tt == TT.NULL:
            self._advance()
            return NullLiteral(line=tok.line, col=tok.col)

        if tt == TT.IDENT:
            self._advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)

        # Keyword identifiers usable as values / function names / keys
        _kw_as_ident = {
            TT.KW_LASER, TT.KW_MIRROR, TT.KW_SENSOR, TT.KW_TENSOR,
            TT.KW_BEAM, TT.KW_SPECTRUM, TT.KW_MATRIX,
            TT.KW_MESH, TT.KW_TILE, TT.KW_CANVAS,
            TT.KW_RANGE, TT.KW_TYPE, TT.KW_DTYPE, TT.KW_GRAPH,
            TT.KW_BATCH, TT.KW_POLYGON, TT.KW_REPLICATED, TT.KW_VISIBLE,
        }
        if tt in _kw_as_ident:
            self._advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)

        if tt == TT.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TT.RPAREN)
            return expr

        if tt == TT.LBRACKET:
            line, col = self._lc()
            self._advance()
            elements: list[Node] = []
            while not self._check(TT.RBRACKET, TT.EOF):
                elements.append(self._parse_expr())
                if not self._match(TT.COMMA):
                    break
            self._expect(TT.RBRACKET)
            return ListLiteral(elements=elements, line=line, col=col)

        if tt == TT.LBRACE:
            line, col = self._lc()
            self._advance()
            pairs: list[tuple[Node, Node]] = []
            while not self._check(TT.RBRACE, TT.EOF):
                k = self._parse_expr()
                self._expect(TT.COLON)
                v = self._parse_expr()
                pairs.append((k, v))
                if not self._match(TT.COMMA):
                    self._skip_semis()
            self._expect(TT.RBRACE)
            return DictLiteral(pairs=pairs, line=line, col=col)

        if tt == TT.KW_SPECTRUM:
            return self._parse_spectrum_expr()

        raise ParseError(
            f"Unexpected token {tt.name} ({tok.value!r}) at line {tok.line}, col {tok.col}"
        )

    def _parse_spectrum_expr(self) -> SpectrumExpr:
        line, col = self._lc()
        self._advance()
        self._expect(TT.LPAREN)
        lo = self._parse_expr()
        self._expect(TT.COMMA)
        hi = self._parse_expr()
        self._expect(TT.RPAREN)
        unit = "nm"
        if self._peek().type == TT.IDENT and self._peek().value in ("nm", "um", "mm"):
            unit = self._advance().value
        return SpectrumExpr(lo=lo, hi=hi, unit=unit, line=line, col=col)

    # ------------------------------------------------------------------
    # Call argument list  ( arg, arg, key=val )
    # ------------------------------------------------------------------

    def _parse_call_args(self) -> tuple[list[Node], dict[str, Node]]:
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        self._expect(TT.LPAREN)
        while not self._check(TT.RPAREN, TT.EOF):
            if self._peek().type == TT.IDENT and self._peek(1).type == TT.EQ:
                k = self._advance().value
                self._advance()  # =
                kwargs[k] = self._parse_expr()
            else:
                args.append(self._parse_expr())
            if not self._match(TT.COMMA):
                break
        self._expect(TT.RPAREN)
        return args, kwargs


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def parse(source: str) -> Program:
    """Tokenize and parse *source*, returning a :class:`Program` AST."""
    tokens = tokenize(source)
    return Parser(tokens).parse()
