"""Documental directory – hierarchical documentation namespace.

A :class:`DocDirectory` mirrors a file-system directory but is entirely
in-memory.  Programs can create nested doc trees, attach metadata, and
export to JSON or plain-text index files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DocEntry:
    """A single documentation entry (file analogue)."""

    def __init__(
        self,
        name: str,
        content: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name:       str = name
        self.content:    str = content
        self.tags:       list[str] = tags or []
        self.metadata:   dict[str, Any] = metadata or {}
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at

    def update(self, content: str) -> None:
        self.content    = content
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "content":    self.content,
            "tags":       self.tags,
            "metadata":   self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __repr__(self) -> str:
        return f"DocEntry(name={self.name!r}, chars={len(self.content)})"


class DocDirectory:
    """Hierarchical documentation directory.

    Parameters
    ----------
    name:
        Directory name.
    parent:
        Optional parent directory.
    """

    def __init__(
        self,
        name: str,
        parent: "DocDirectory | None" = None,
    ) -> None:
        self.name:     str = name
        self.parent:   "DocDirectory | None" = parent
        self._entries: dict[str, DocEntry] = {}
        self._subdirs: dict[str, "DocDirectory"] = {}
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(self, name: str, content: str = "",
                  tags: list[str] | None = None,
                  **metadata: Any) -> DocEntry:
        entry = DocEntry(name=name, content=content, tags=tags or [],
                         metadata=metadata)
        self._entries[name] = entry
        return entry

    def get_entry(self, name: str) -> DocEntry | None:
        return self._entries.get(name)

    def remove_entry(self, name: str) -> None:
        self._entries.pop(name, None)

    @property
    def entries(self) -> list[DocEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Sub-directory management
    # ------------------------------------------------------------------

    def mkdir(self, name: str) -> "DocDirectory":
        sub = DocDirectory(name=name, parent=self)
        self._subdirs[name] = sub
        return sub

    def cd(self, name: str) -> "DocDirectory":
        if name not in self._subdirs:
            raise KeyError(f"Sub-directory {name!r} not found in {self.name!r}")
        return self._subdirs[name]

    def get_or_create(self, name: str) -> "DocDirectory":
        if name not in self._subdirs:
            return self.mkdir(name)
        return self._subdirs[name]

    @property
    def subdirs(self) -> list["DocDirectory"]:
        return list(self._subdirs.values())

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        if self.parent is None:
            return self.name
        return f"{self.parent.path}/{self.name}"

    def resolve_path(self, path: str) -> "DocDirectory | DocEntry | None":
        """Navigate to a path relative to this directory."""
        parts = path.strip("/").split("/")
        current: DocDirectory | DocEntry | None = self
        for part in parts:
            if isinstance(current, DocDirectory):
                if part in current._subdirs:
                    current = current._subdirs[part]
                elif part in current._entries:
                    current = current._entries[part]
                else:
                    return None
            else:
                return None
        return current

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, deep: bool = True) -> list[DocEntry]:
        """Find entries whose name or content contains *query*."""
        query_lower = query.lower()
        results: list[DocEntry] = []
        for entry in self._entries.values():
            if (query_lower in entry.name.lower()
                    or query_lower in entry.content.lower()
                    or any(query_lower in t.lower() for t in entry.tags)):
                results.append(entry)
        if deep:
            for sub in self._subdirs.values():
                results.extend(sub.search(query, deep=True))
        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name":    self.name,
            "path":    self.path,
            "entries": {n: e.to_dict() for n, e in self._entries.items()},
            "subdirs": {n: s.to_dict() for n, s in self._subdirs.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def tree(self, indent: int = 0) -> str:
        """Return an ASCII directory tree."""
        prefix = "  " * indent
        lines = [f"{prefix}{self.name}/"]
        for e in self._entries.values():
            lines.append(f"{prefix}  {e.name}")
        for s in self._subdirs.values():
            lines.append(s.tree(indent + 1))
        return "\n".join(lines)

    def save_to_disk(self, base_path: str) -> None:
        """Persist the directory tree to the file system."""
        p = Path(base_path) / self.name
        p.mkdir(parents=True, exist_ok=True)
        for entry in self._entries.values():
            (p / f"{entry.name}.md").write_text(entry.content, encoding="utf-8")
        for sub in self._subdirs.values():
            sub.save_to_disk(str(p))

    def __repr__(self) -> str:
        return (
            f"DocDirectory(name={self.name!r}, "
            f"entries={len(self._entries)}, "
            f"subdirs={len(self._subdirs)})"
        )
