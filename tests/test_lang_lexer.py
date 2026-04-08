"""Tests for the HoloLang lexer."""

import pytest
from hololang.lang.lexer import tokenize, TT, Token


def test_integer_literal():
    tokens = tokenize("42")
    assert tokens[0].type == TT.INTEGER
    assert tokens[0].value == 42


def test_float_literal():
    tokens = tokenize("3.14")
    assert tokens[0].type == TT.FLOAT
    assert abs(tokens[0].value - 3.14) < 1e-9


def test_hex_integer():
    tokens = tokenize("0xFF")
    assert tokens[0].value == 255


def test_binary_integer():
    tokens = tokenize("0b1010")
    assert tokens[0].value == 10


def test_string_literal():
    tokens = tokenize('"hello world"')
    assert tokens[0].type == TT.STRING
    assert tokens[0].value == "hello world"


def test_bool_true():
    tokens = tokenize("true")
    assert tokens[0].type == TT.BOOL
    assert tokens[0].value is True


def test_bool_false():
    tokens = tokenize("false")
    assert tokens[0].type == TT.BOOL
    assert tokens[0].value is False


def test_null_literal():
    tokens = tokenize("null")
    assert tokens[0].type == TT.NULL


def test_keywords():
    kw_map = {
        "device": TT.KW_DEVICE,
        "laser":  TT.KW_LASER,
        "tensor": TT.KW_TENSOR,
        "mesh":   TT.KW_MESH,
        "enum":   TT.KW_ENUM,
        "kernel": TT.KW_KERNEL,
        "pool":   TT.KW_POOL,
        "fn":     TT.KW_FN,
        "if":     TT.KW_IF,
        "while":  TT.KW_WHILE,
    }
    for word, expected_tt in kw_map.items():
        tokens = tokenize(word)
        assert tokens[0].type == expected_tt, f"Failed for keyword {word!r}"


def test_operators():
    source = "-> => <- := == != <= >= .."
    tokens = tokenize(source)
    types = [t.type for t in tokens if t.type != TT.EOF]
    expected = [TT.ARROW, TT.FAT_ARROW, TT.LEFT_ARROW, TT.COLON_EQ,
                TT.EQEQ, TT.NEQ, TT.LTE, TT.GTE, TT.DOTDOT]
    assert types == expected


def test_line_comment():
    tokens = tokenize("42 // this is a comment\n99")
    types = [t.type for t in tokens if t.type != TT.NEWLINE and t.type != TT.EOF]
    values = [t.value for t in tokens if t.type == TT.INTEGER]
    assert values == [42, 99]


def test_block_comment():
    tokens = tokenize("/* block */ 77")
    ints = [t for t in tokens if t.type == TT.INTEGER]
    assert ints[0].value == 77


def test_hash_comment():
    tokens = tokenize("# hash comment\n55")
    ints = [t for t in tokens if t.type == TT.INTEGER]
    assert ints[0].value == 55


def test_line_col_tracking():
    tokens = tokenize("a\nb")
    assert tokens[0].line == 1
    # b is on line 2
    ident_tokens = [t for t in tokens if t.type == TT.IDENT]
    assert ident_tokens[0].line == 1
    assert ident_tokens[1].line == 2


def test_nm_wavelength_ident_present():
    """532nm – the integer 532 followed by the identifier 'nm'."""
    tokens = tokenize("532nm")
    assert tokens[0].type == TT.INTEGER
    assert tokens[0].value == 532
    assert tokens[1].value == "nm"


def test_unknown_char_raises():
    with pytest.raises(SyntaxError):
        tokenize("$invalid")


def test_unterminated_block_comment_raises():
    with pytest.raises(SyntaxError):
        tokenize("/* unterminated")


def test_eof_token():
    tokens = tokenize("")
    assert tokens[-1].type == TT.EOF
