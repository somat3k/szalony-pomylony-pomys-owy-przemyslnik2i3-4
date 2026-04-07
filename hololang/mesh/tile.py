"""Mesh tile – the atomic unit of the holographic canvas MDI.

Each tile occupies a (row, col) cell in the canvas grid and holds:
* A data payload (tensor slice, device state, scalar, …)
* A list of outgoing impulse connections
* Rendering metadata (colour, label, …)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TileStyle:
    """Visual style for canvas rendering."""
    color:      str = "#1e90ff"
    text_color: str = "#ffffff"
    border:     str = "#0050a0"
    label:      str = ""
    icon:       str = ""


class Tile:
    """A single cell in the mesh canvas.

    Parameters
    ----------
    row, col:
        Zero-based grid position.
    data:
        Arbitrary payload stored in this tile.
    style:
        Optional :class:`TileStyle` for rendering.
    """

    def __init__(
        self,
        row: int,
        col: int,
        data: Any = None,
        style: TileStyle | None = None,
    ) -> None:
        self.row    = row
        self.col    = col
        self.data   = data
        self.style  = style or TileStyle()

        # Impulse connections: list of (target_tile_key, transform_fn)
        self._connections: list[tuple[tuple[int, int], Callable | None]] = []
        self._received_impulses: list[Any] = []

    # ------------------------------------------------------------------
    # Connections / impulses
    # ------------------------------------------------------------------

    def connect_to(
        self,
        target_row: int,
        target_col: int,
        transform: Callable | None = None,
    ) -> None:
        """Register an outgoing impulse edge to another tile."""
        self._connections.append(((target_row, target_col), transform))

    def send_impulse(self, payload: Any, canvas: "Any") -> int:
        """Fire all outgoing impulses, routing *payload* to connected tiles.

        Returns
        -------
        int
            Number of impulses actually delivered to downstream tiles.
        """
        count = 0
        for (tr, tc), fn in self._connections:
            value = fn(payload) if fn is not None else payload
            target = canvas.get_tile(tr, tc)
            if target is not None:
                target.receive_impulse(value)
                count += 1
        return count

    def receive_impulse(self, payload: Any) -> None:
        self._received_impulses.append(payload)

    def flush_impulses(self) -> list[Any]:
        impulses = list(self._received_impulses)
        self._received_impulses.clear()
        return impulses

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Tile({self.row}, {self.col}, data={self.data!r})"

    def to_dict(self) -> dict:
        return {
            "row":   self.row,
            "col":   self.col,
            "data":  str(self.data),
            "label": self.style.label,
            "color": self.style.color,
            "connections": [
                {"row": r, "col": c} for (r, c), _ in self._connections
            ],
        }
