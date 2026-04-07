"""Multi-document interface (MDI) canvas for holographic mesh tiles.

The canvas is an infinite (sparse) 2-D grid of :class:`~hololang.mesh.tile.Tile`
objects.  It supports:
* Adding / querying tiles
* Running impulse cycles across the whole grid
* Rendering to ASCII art or JSON for debugging
* Binding to a tensor or device source
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from hololang.mesh.tile import Tile, TileStyle


class Canvas:
    """MDI mesh canvas.

    Parameters
    ----------
    name:
        Human-readable name.
    source:
        Optional backing object (Tensor, Device, …).
    """

    def __init__(self, name: str = "canvas", source: Any = None) -> None:
        self.name   = name
        self.source = source
        self._tiles: dict[tuple[int, int], Tile] = {}
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Tile management
    # ------------------------------------------------------------------

    def add_tile(self, tile: Tile) -> None:
        self._tiles[(tile.row, tile.col)] = tile

    def put(self, row: int, col: int, data: Any = None,
            style: TileStyle | None = None) -> Tile:
        tile = Tile(row=row, col=col, data=data, style=style)
        self._tiles[(row, col)] = tile
        return tile

    def get_tile(self, row: int, col: int) -> Tile | None:
        return self._tiles.get((row, col))

    def remove_tile(self, row: int, col: int) -> None:
        self._tiles.pop((row, col), None)

    def __iter__(self) -> Iterator[Tile]:
        return iter(self._tiles.values())

    def __len__(self) -> int:
        return len(self._tiles)

    # ------------------------------------------------------------------
    # Impulse cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> int:
        """Fire one impulse propagation cycle across all tiles.

        Returns
        -------
        int
            Total number of impulses actually delivered to downstream tiles.
        """
        dispatched = 0
        # Snapshot impulse queues BEFORE processing to avoid counting
        # impulses that arrive during this same cycle.
        snapshot = {key: tile.flush_impulses()
                    for key, tile in list(self._tiles.items())}
        for (row, col), impulses in snapshot.items():
            tile = self._tiles.get((row, col))
            if tile is None:
                continue
            for impulse in impulses:
                dispatched += tile.send_impulse(impulse, self)
        self._cycle_count += 1
        return dispatched

    def send_impulse(self, from_row: int, from_col: int,
                     to_row: int, to_col: int,
                     payload: Any) -> None:
        """Manually send a single impulse between two tiles."""
        src = self.get_tile(from_row, from_col)
        dst = self.get_tile(to_row, to_col)
        if dst is not None:
            value = payload
            if src is not None:
                for (tr, tc), fn in src._connections:
                    if (tr, tc) == (to_row, to_col) and fn is not None:
                        value = fn(payload)
            dst.receive_impulse(value)

    # ------------------------------------------------------------------
    # Config / parameter helpers (called from HoloLang programs)
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_ascii(self, cell_width: int = 6) -> str:
        """Render the canvas as ASCII art for terminal debugging."""
        if not self._tiles:
            return f"[Canvas '{self.name}' – empty]"
        rows = [r for (r, _) in self._tiles]
        cols = [c for (_, c) in self._tiles]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)

        lines = [f"Canvas: {self.name}"]
        header = "    " + "".join(f"{c:<{cell_width}}" for c in range(c_min, c_max + 1))
        lines.append(header)
        lines.append("    " + "-" * ((c_max - c_min + 1) * cell_width))

        for r in range(r_min, r_max + 1):
            row_cells = []
            for c in range(c_min, c_max + 1):
                tile = self.get_tile(r, c)
                if tile is None:
                    cell = "·" * (cell_width - 1)
                else:
                    label = tile.style.label or str(tile.data)[:cell_width - 1]
                    cell = label.ljust(cell_width - 1)
                row_cells.append(cell)
            lines.append(f"{r:3} |" + " ".join(row_cells))
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "name":   self.name,
                "tiles":  [t.to_dict() for t in self._tiles.values()],
                "cycles": self._cycle_count,
            },
            indent=2,
        )

    def __repr__(self) -> str:
        return f"Canvas(name={self.name!r}, tiles={len(self._tiles)}, cycles={self._cycle_count})"
