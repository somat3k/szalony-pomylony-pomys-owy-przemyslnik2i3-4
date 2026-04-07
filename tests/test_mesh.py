"""Tests for the mesh / canvas / tile subsystem."""

import pytest
from hololang.mesh.tile import Tile, TileStyle
from hololang.mesh.canvas import Canvas
from hololang.mesh.display import Display


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

def test_tile_create():
    t = Tile(0, 0, data=42)
    assert t.row == 0
    assert t.col == 0
    assert t.data == 42


def test_tile_connect_impulse():
    canvas = Canvas("test")
    src = canvas.put(0, 0, data=10)
    dst = canvas.put(0, 1, data=0)
    src.connect_to(0, 1)

    # Manually queue an impulse in src (simulating canvas.run_cycle sending)
    src.receive_impulse(99)
    src.send_impulse(99, canvas)
    received = dst.flush_impulses()
    assert 99 in received


def test_tile_connect_with_transform():
    canvas = Canvas("transform_test")
    src = canvas.put(1, 0, data="input")
    dst = canvas.put(1, 1, data=None)
    src.connect_to(1, 1, transform=lambda x: x * 2)

    src.receive_impulse(5)
    src.send_impulse(5, canvas)
    received = dst.flush_impulses()
    assert 10 in received


def test_tile_flush():
    t = Tile(0, 0)
    t.receive_impulse("a")
    t.receive_impulse("b")
    flushed = t.flush_impulses()
    assert flushed == ["a", "b"]
    # After flush, buffer is empty
    assert t.flush_impulses() == []


def test_tile_to_dict():
    t = Tile(2, 3, data="hello", style=TileStyle(label="HI"))
    d = t.to_dict()
    assert d["row"] == 2
    assert d["col"] == 3
    assert d["label"] == "HI"


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

def test_canvas_add_get_tile():
    c = Canvas("mycanvas")
    t = c.put(0, 0, data="test")
    assert c.get_tile(0, 0) is t


def test_canvas_missing_tile():
    c = Canvas("empty")
    assert c.get_tile(99, 99) is None


def test_canvas_len():
    c = Canvas("sized")
    c.put(0, 0)
    c.put(1, 0)
    c.put(0, 1)
    assert len(c) == 3


def test_canvas_remove_tile():
    c = Canvas("remove")
    c.put(5, 5)
    assert c.get_tile(5, 5) is not None
    c.remove_tile(5, 5)
    assert c.get_tile(5, 5) is None


def test_canvas_run_cycle_dispatches():
    c = Canvas("cycle")
    src = c.put(0, 0)
    dst = c.put(0, 1)
    src.connect_to(0, 1)
    # Queue an impulse in src
    src.receive_impulse("ping")
    dispatched = c.run_cycle()
    assert dispatched == 1
    assert "ping" in dst.flush_impulses()


def test_canvas_send_impulse():
    c = Canvas("si")
    c.put(0, 0)
    c.put(0, 1)
    c.send_impulse(0, 0, 0, 1, "payload")
    # Directly check dst received it
    dst = c.get_tile(0, 1)
    assert "payload" in dst.flush_impulses()


def test_canvas_cycle_count():
    c = Canvas("cc")
    c.put(0, 0)
    c.run_cycle()
    c.run_cycle()
    assert c._cycle_count == 2


def test_canvas_iter():
    c = Canvas("iter")
    c.put(0, 0)
    c.put(1, 1)
    tiles = list(c)
    assert len(tiles) == 2


def test_canvas_to_json():
    c = Canvas("json_test")
    c.put(0, 0, data="hello")
    j = c.to_json()
    assert "json_test" in j
    assert "hello" in j


def test_canvas_ascii():
    c = Canvas("ascii")
    c.put(0, 0, data="A")
    c.put(0, 1, data="B")
    art = c.to_ascii()
    assert "ascii" in art


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def test_display_terminal_empty():
    c = Canvas("empty_canvas")
    d = Display(c, use_color=False)
    result = d.render_terminal()
    assert "empty" in result.lower()


def test_display_terminal():
    c = Canvas("display_test")
    c.put(0, 0, data="R", style=TileStyle(label="R", color="#ff0000"))
    c.put(0, 1, data="G", style=TileStyle(label="G", color="#00ff00"))
    d = Display(c, use_color=False)
    result = d.render_terminal()
    assert "R" in result
    assert "G" in result


def test_display_json():
    c = Canvas("json_disp")
    c.put(0, 0)
    d = Display(c)
    j = d.render_json()
    assert "json_disp" in j


def test_display_svg():
    c = Canvas("svg_disp")
    c.put(0, 0, data="X", style=TileStyle(color="#0000ff"))
    c.put(0, 1, data="Y")
    c.get_tile(0, 0).connect_to(0, 1)
    d = Display(c)
    svg = d.render_svg()
    assert "<svg" in svg
    assert "svg_disp" in svg


def test_display_svg_empty():
    c = Canvas("empty_svg")
    d = Display(c)
    svg = d.render_svg()
    assert "<svg" in svg
