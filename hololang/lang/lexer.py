"""HoloLang lexer – converts source text into a flat token stream."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TT(Enum):
    """Token type enumeration."""
    # Literals
    INTEGER   = auto()
    FLOAT     = auto()
    STRING    = auto()
    BOOL      = auto()
    NULL      = auto()

    # Identifiers & keywords
    IDENT     = auto()
    KW_DEVICE   = auto()
    KW_LASER    = auto()
    KW_MIRROR   = auto()
    KW_SENSOR   = auto()
    KW_TENSOR   = auto()
    KW_SAFE     = auto()
    KW_MESH     = auto()
    KW_TILE     = auto()
    KW_CANVAS   = auto()
    KW_KERNEL   = auto()
    KW_POOL     = auto()
    KW_RUNTIME  = auto()
    KW_SESSION  = auto()
    KW_SKILL    = auto()
    KW_DOC      = auto()
    KW_ENUM     = auto()
    KW_INVOKE   = auto()
    KW_TRANSFORM = auto()
    KW_BEAM     = auto()
    KW_SPECTRUM = auto()
    KW_MATRIX   = auto()
    KW_CHANNEL  = auto()
    KW_WEBHOOK  = auto()
    KW_API      = auto()
    KW_IMPORT   = auto()
    KW_LET      = auto()
    KW_CONST    = auto()
    KW_FN       = auto()
    KW_RETURN   = auto()
    KW_IF       = auto()
    KW_ELSE     = auto()
    KW_FOR      = auto()
    KW_IN       = auto()
    KW_WHILE    = auto()
    KW_BREAK    = auto()
    KW_CONTINUE = auto()
    KW_TRUE     = auto()
    KW_FALSE    = auto()
    KW_NULL     = auto()
    KW_TYPE     = auto()
    KW_DTYPE    = auto()
    KW_REPLICATED = auto()
    KW_VISIBLE  = auto()
    KW_RANGE    = auto()
    KW_POLYGON  = auto()
    KW_BATCH    = auto()
    KW_GRAPH    = auto()
    KW_DEBUG    = auto()
    KW_IMPULSE  = auto()
    KW_CONFIG   = auto()
    KW_PARAM    = auto()
    KW_EMIT     = auto()
    KW_LISTEN   = auto()
    KW_CONNECT  = auto()
    KW_BIND     = auto()
    KW_POOL_RUN = auto()

    # Operators
    ARROW       = auto()   # ->
    FAT_ARROW   = auto()   # =>
    LEFT_ARROW  = auto()   # <-
    COLON_EQ    = auto()   # :=
    EQ          = auto()   # =
    EQEQ        = auto()   # ==
    NEQ         = auto()   # !=
    LT          = auto()   # <
    GT          = auto()   # >
    LTE         = auto()   # <=
    GTE         = auto()   # >=
    PLUS        = auto()
    MINUS       = auto()
    STAR        = auto()
    SLASH       = auto()
    PERCENT     = auto()
    AMPERSAND   = auto()
    PIPE        = auto()
    CARET       = auto()
    TILDE       = auto()
    BANG        = auto()
    AT          = auto()   # @  (decorator / annotation)
    DOT         = auto()
    DOTDOT      = auto()   # ..
    COLON       = auto()
    SEMICOLON   = auto()
    COMMA       = auto()
    HASH        = auto()   # # (inline comment start – handled in lexer)

    # Delimiters
    LPAREN      = auto()
    RPAREN      = auto()
    LBRACE      = auto()
    RBRACE      = auto()
    LBRACKET    = auto()
    RBRACKET    = auto()

    # Special
    NEWLINE     = auto()
    EOF         = auto()


# ---------------------------------------------------------------------------
# Keywords map
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, TT] = {
    "device":     TT.KW_DEVICE,
    "laser":      TT.KW_LASER,
    "mirror":     TT.KW_MIRROR,
    "sensor":     TT.KW_SENSOR,
    "tensor":     TT.KW_TENSOR,
    "safe":       TT.KW_SAFE,
    "mesh":       TT.KW_MESH,
    "tile":       TT.KW_TILE,
    "canvas":     TT.KW_CANVAS,
    "kernel":     TT.KW_KERNEL,
    "pool":       TT.KW_POOL,
    "runtime":    TT.KW_RUNTIME,
    "session":    TT.KW_SESSION,
    "skill":      TT.KW_SKILL,
    "doc":        TT.KW_DOC,
    "enum":       TT.KW_ENUM,
    "invoke":     TT.KW_INVOKE,
    "transform":  TT.KW_TRANSFORM,
    "beam":       TT.KW_BEAM,
    "spectrum":   TT.KW_SPECTRUM,
    "matrix":     TT.KW_MATRIX,
    "channel":    TT.KW_CHANNEL,
    "webhook":    TT.KW_WEBHOOK,
    "api":        TT.KW_API,
    "import":     TT.KW_IMPORT,
    "let":        TT.KW_LET,
    "const":      TT.KW_CONST,
    "fn":         TT.KW_FN,
    "return":     TT.KW_RETURN,
    "if":         TT.KW_IF,
    "else":       TT.KW_ELSE,
    "for":        TT.KW_FOR,
    "in":         TT.KW_IN,
    "while":      TT.KW_WHILE,
    "break":      TT.KW_BREAK,
    "continue":   TT.KW_CONTINUE,
    "true":       TT.KW_TRUE,
    "false":      TT.KW_FALSE,
    "null":       TT.KW_NULL,
    "type":       TT.KW_TYPE,
    "dtype":      TT.KW_DTYPE,
    "replicated": TT.KW_REPLICATED,
    "visible":    TT.KW_VISIBLE,
    "range":      TT.KW_RANGE,
    "polygon":    TT.KW_POLYGON,
    "batch":      TT.KW_BATCH,
    "graph":      TT.KW_GRAPH,
    "debug":      TT.KW_DEBUG,
    "impulse":    TT.KW_IMPULSE,
    "config":     TT.KW_CONFIG,
    "param":      TT.KW_PARAM,
    "emit":       TT.KW_EMIT,
    "listen":     TT.KW_LISTEN,
    "connect":    TT.KW_CONNECT,
    "bind":       TT.KW_BIND,
    "pool_run":   TT.KW_POOL_RUN,
}


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: TT
    value: object
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_IDENT_RE      = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Float regex: decimal form (digits.digits, NOT followed by another dot)
# OR scientific notation (digits[eE]±digits).  The negative lookahead (?!\.)
# prevents "0..10" from being consumed as float 0.0 followed by member access.
_FLOAT_RE      = re.compile(r"\d+\.(?!\.)(\d*)([eE][+-]?\d+)?(?=\D|$)|\d+[eE][+-]?\d+")
_INT_RE        = re.compile(r"0x[0-9A-Fa-f]+|0b[01]+|0o[0-7]+|\d+")
_STRING_RE     = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_WHITESPACE_RE = re.compile(r"[ \t\r]+")
_NEWLINE_RE    = re.compile(r"\n")


def tokenize(source: str) -> list[Token]:
    """Tokenize *source* text and return a list of :class:`Token` objects."""
    tokens: list[Token] = []
    pos = 0
    line = 1
    line_start = 0

    def col() -> int:
        return pos - line_start + 1

    while pos < len(source):
        # Skip block comments   /* ... */
        if source[pos:pos+2] == "/*":
            end = source.find("*/", pos + 2)
            if end == -1:
                raise SyntaxError(f"Unterminated block comment at line {line}")
            newlines = source[pos:end+2].count("\n")
            line += newlines
            if newlines:
                last_nl = source.rfind("\n", pos, end + 2)
                line_start = last_nl + 1
            pos = end + 2
            continue

        # Skip line comments  // ...
        if source[pos:pos+2] == "//":
            end = source.find("\n", pos)
            pos = end if end != -1 else len(source)
            continue

        # Skip line comments  # ...
        if source[pos] == "#":
            end = source.find("\n", pos)
            pos = end if end != -1 else len(source)
            continue

        # Whitespace
        m = _WHITESPACE_RE.match(source, pos)
        if m:
            pos = m.end()
            continue

        # Newline
        m = _NEWLINE_RE.match(source, pos)
        if m:
            tokens.append(Token(TT.NEWLINE, "\n", line, col()))
            pos = m.end()
            line += 1
            line_start = pos
            continue

        # String literals
        m = _STRING_RE.match(source, pos)
        if m:
            raw = m.group()
            # Unescape simple escapes
            value = raw[1:-1].encode("raw_unicode_escape").decode("unicode_escape")
            tokens.append(Token(TT.STRING, value, line, col()))
            pos = m.end()
            continue

        # Float before int (longer match)
        m = _FLOAT_RE.match(source, pos)
        if m:
            tokens.append(Token(TT.FLOAT, float(m.group()), line, col()))
            pos = m.end()
            continue

        # Integer
        m = _INT_RE.match(source, pos)
        if m:
            raw = m.group()
            if raw.startswith("0x"):
                value = int(raw, 16)
            elif raw.startswith("0b"):
                value = int(raw, 2)
            elif raw.startswith("0o"):
                value = int(raw, 8)
            else:
                value = int(raw)
            tokens.append(Token(TT.INTEGER, value, line, col()))
            pos = m.end()
            continue

        # Identifier / keyword
        m = _IDENT_RE.match(source, pos)
        if m:
            word = m.group()
            tt = KEYWORDS.get(word, TT.IDENT)
            # Map true/false/null to literal token types
            if tt == TT.KW_TRUE:
                tokens.append(Token(TT.BOOL, True, line, col()))
            elif tt == TT.KW_FALSE:
                tokens.append(Token(TT.BOOL, False, line, col()))
            elif tt == TT.KW_NULL:
                tokens.append(Token(TT.NULL, None, line, col()))
            else:
                tokens.append(Token(tt, word, line, col()))
            pos = m.end()
            continue

        # Multi-char operators (longest match first)
        c2 = source[pos:pos+2]
        multi = {
            "->": TT.ARROW,
            "=>": TT.FAT_ARROW,
            "<-": TT.LEFT_ARROW,
            ":=": TT.COLON_EQ,
            "==": TT.EQEQ,
            "!=": TT.NEQ,
            "<=": TT.LTE,
            ">=": TT.GTE,
            "..": TT.DOTDOT,
        }
        if c2 in multi:
            tokens.append(Token(multi[c2], c2, line, col()))
            pos += 2
            continue

        # Single-char operators / delimiters
        single = {
            "=": TT.EQ, "<": TT.LT, ">": TT.GT, "+": TT.PLUS,
            "-": TT.MINUS, "*": TT.STAR, "/": TT.SLASH,
            "%": TT.PERCENT, "&": TT.AMPERSAND, "|": TT.PIPE,
            "^": TT.CARET, "~": TT.TILDE, "!": TT.BANG,
            "@": TT.AT, ".": TT.DOT, ":": TT.COLON,
            ";": TT.SEMICOLON, ",": TT.COMMA,
            "(": TT.LPAREN, ")": TT.RPAREN,
            "{": TT.LBRACE, "}": TT.RBRACE,
            "[": TT.LBRACKET, "]": TT.RBRACKET,
        }
        ch = source[pos]
        if ch in single:
            tokens.append(Token(single[ch], ch, line, col()))
            pos += 1
            continue

        raise SyntaxError(
            f"Unexpected character {source[pos]!r} at line {line}, col {col()}"
        )

    tokens.append(Token(TT.EOF, None, line, col()))
    return tokens
