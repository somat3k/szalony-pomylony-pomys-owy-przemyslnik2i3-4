"""Canvas display driver.

Renders a :class:`~hololang.mesh.canvas.Canvas` to different output targets:
* Terminal (ANSI colour)
* Plain text
* JSON
* SVG (vector graphics for documentation)
"""

from __future__ import annotations

import json
from typing import Any

from hololang.mesh.canvas import Canvas


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD  = "\033[1m"


def _ansi_fg(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ANSI 24-bit foreground escape."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


def _ansi_bg(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ANSI 24-bit background escape."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"\033[48;2;{r};{g};{b}m"


# ---------------------------------------------------------------------------
# Display class
# ---------------------------------------------------------------------------

class Display:
    """Renders a :class:`Canvas` to various output formats.

    Parameters
    ----------
    canvas:
        The canvas to render.
    cell_width:
        Character width of each tile cell in text outputs.
    use_color:
        Enable ANSI color output in terminal renders.
    """

    def __init__(
        self,
        canvas: Canvas,
        cell_width: int = 8,
        use_color: bool = True,
    ) -> None:
        self.canvas     = canvas
        self.cell_width = cell_width
        self.use_color  = use_color

    # ------------------------------------------------------------------
    # Terminal render
    # ------------------------------------------------------------------

    def render_terminal(self) -> str:
        """Return an ANSI-colored terminal representation."""
        if not self.canvas._tiles:
            return f"[empty canvas: {self.canvas.name}]"

        rows = sorted({r for (r, _) in self.canvas._tiles})
        cols = sorted({c for (_, c) in self.canvas._tiles})

        lines: list[str] = []
        title = f" {self.canvas.name} "
        width = len(cols) * (self.cell_width + 1) + 4
        lines.append(_ANSI_BOLD + title.center(width, "═") + _ANSI_RESET)

        for row in rows:
            row_parts: list[str] = []
            for col in cols:
                tile = self.canvas.get_tile(row, col)
                if tile is None:
                    cell = " " * self.cell_width
                    row_parts.append(cell)
                else:
                    label = tile.style.label or str(tile.data)
                    label = label[: self.cell_width - 2].center(self.cell_width - 2)
                    if self.use_color:
                        cell = (
                            _ansi_bg(tile.style.color)
                            + _ansi_fg(tile.style.text_color)
                            + f"[{label}]"
                            + _ANSI_RESET
                        )
                    else:
                        cell = f"[{label}]"
                    row_parts.append(cell)
            lines.append(" ".join(row_parts))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON render
    # ------------------------------------------------------------------

    def render_json(self, indent: int = 2) -> str:
        return self.canvas.to_json(indent=indent)

    # ------------------------------------------------------------------
    # SVG render
    # ------------------------------------------------------------------

    def render_svg(
        self,
        tile_px: int = 80,
        font_size: int = 12,
        padding: int = 10,
    ) -> str:
        """Render the canvas as an SVG document."""
        if not self.canvas._tiles:
            return "<svg xmlns='http://www.w3.org/2000/svg'/>"

        rows = sorted({r for (r, _) in self.canvas._tiles})
        cols = sorted({c for (_, c) in self.canvas._tiles})

        width  = (max(cols) + 1) * tile_px + 2 * padding
        height = (max(rows) + 1) * tile_px + 2 * padding

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}">',
            f'<defs>'
            f'<marker id="arr" markerWidth="8" markerHeight="8" '
            f'refX="6" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="#ffaa00"/>'
            f'</marker>'
            f'</defs>',
            f'<rect width="{width}" height="{height}" fill="#111"/>',
            f'<text x="{padding}" y="{padding + font_size}" '
            f'font-family="monospace" font-size="{font_size}" fill="#aaa">'
            f'{self.canvas.name}</text>',
        ]

        for (row, col), tile in self.canvas._tiles.items():
            x = padding + col * tile_px
            y = padding + row * tile_px
            label = (tile.style.label or str(tile.data))[:12]
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{tile_px - 2}" '
                f'height="{tile_px - 2}" rx="4" fill="{tile.style.color}" '
                f'stroke="{tile.style.border}" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{x + tile_px // 2}" y="{y + tile_px // 2 + font_size // 2}" '
                f'font-family="monospace" font-size="{font_size}" '
                f'fill="{tile.style.text_color}" text-anchor="middle">'
                f'{label}</text>'
            )

        # Draw impulse connections
        for (row, col), tile in self.canvas._tiles.items():
            for (tr, tc), _ in tile._connections:
                x1 = padding + col * tile_px + tile_px // 2
                y1 = padding + row * tile_px + tile_px // 2
                x2 = padding + tc  * tile_px + tile_px // 2
                y2 = padding + tr  * tile_px + tile_px // 2
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="#ffaa00" stroke-width="1.5" '
                    f'marker-end="url(#arr)"/>'
                )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
