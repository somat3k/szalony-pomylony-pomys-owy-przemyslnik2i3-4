"""HoloLang command-line interface.

Usage::

    hololang run program.hl
    hololang repl
    hololang check program.hl
    hololang info
    hololang skills
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    from hololang.runtime import HoloRuntime
    with HoloRuntime(session_name=Path(args.file).stem) as rt:
        try:
            rt.run_file(args.file)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Runtime error: {exc}", file=sys.stderr)
            if args.traceback:
                import traceback
                traceback.print_exc()
            return 1
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from hololang.lang.parser import parse
    source = Path(args.file).read_text(encoding="utf-8")
    try:
        ast = parse(source)
        print(f"✓ Parsed successfully – {len(ast.body)} top-level nodes")
        return 0
    except Exception as exc:
        print(f"✗ Parse error: {exc}", file=sys.stderr)
        return 1


def cmd_repl(_args: argparse.Namespace) -> int:
    """Interactive read-eval-print loop."""
    from hololang.runtime import HoloRuntime

    print("HoloLang REPL  (type 'exit' or Ctrl-D to quit)")
    print("─" * 50)
    rt = HoloRuntime(session_name="repl")
    try:
        while True:
            try:
                line = input("hl> ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if line.strip().lower() in ("exit", "quit", ":q"):
                print("Bye!")
                break
            if not line.strip():
                continue
            try:
                rt.run_string(line)
            except Exception as exc:
                print(f"Error: {exc}")
    finally:
        rt.close()
    return 0


def cmd_info(_args: argparse.Namespace) -> int:
    from hololang import __version__
    print(f"HoloLang v{__version__}")
    print("  Custom DSL for holographic device control, tensor processing,")
    print("  and light-manipulation automation.")
    print()
    print("Subsystems:")
    modules = [
        ("hololang.lang",    "Language front-end (lexer / parser / interpreter)"),
        ("hololang.tensor",  "Tensor / SafeTensor / computation graph"),
        ("hololang.device",  "Holographic devices (laser, mirror, sensor)"),
        ("hololang.mesh",    "MDI canvas with mesh tiles and impulse cycles"),
        ("hololang.vm",      "Replicable kernel VM and block controller"),
        ("hololang.network", "gRPC / WebSocket / webhook / REST API"),
        ("hololang.docs",    "Session, skills, and doc directory management"),
    ]
    for mod, desc in modules:
        print(f"  {mod:<26} {desc}")
    return 0


def cmd_skills(_args: argparse.Namespace) -> int:
    from hololang.docs.skills import get_registry
    reg = get_registry()
    skills = sorted(reg.all_skills(), key=lambda s: (s.category, s.name))
    print(f"{'Category':<20} {'Name':<28} {'Level':<15} Description")
    print("─" * 90)
    for s in skills:
        print(f"{s.category:<20} {s.name:<28} {s.level_name:<15} {s.description}")
    return 0


def cmd_canvas(args: argparse.Namespace) -> int:
    """Run a HoloLang file and display the resulting canvas."""
    from hololang.runtime import HoloRuntime
    from hololang.mesh.display import Display
    with HoloRuntime(session_name="canvas_view") as rt:
        try:
            rt.run_file(args.file)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        disp = Display(rt.canvas, use_color=not args.no_color)
        print(disp.render_terminal())
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hololang",
        description="HoloLang – holographic device DSL runtime",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # run
    p_run = sub.add_parser("run", help="Execute a .hl program")
    p_run.add_argument("file", metavar="FILE", help="Path to .hl file")
    p_run.add_argument("--traceback", action="store_true",
                       help="Print full traceback on error")

    # check
    p_check = sub.add_parser("check", help="Syntax-check a .hl program")
    p_check.add_argument("file", metavar="FILE")

    # repl
    sub.add_parser("repl", help="Interactive REPL")

    # info
    sub.add_parser("info", help="Show version and subsystem info")

    # skills
    sub.add_parser("skills", help="List available skills")

    # canvas
    p_canvas = sub.add_parser("canvas", help="Run a program and render its canvas")
    p_canvas.add_argument("file", metavar="FILE")
    p_canvas.add_argument("--no-color", action="store_true")

    args = parser.parse_args(argv)

    handlers = {
        "run":    cmd_run,
        "check":  cmd_check,
        "repl":   cmd_repl,
        "info":   cmd_info,
        "skills": cmd_skills,
        "canvas": cmd_canvas,
    }

    if args.command is None:
        parser.print_help()
        return 0

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
