"""Session management for HoloLang programs.

A :class:`Session` tracks the execution context, timing, inputs, and
artefacts produced during a HoloLang run.  Sessions can be serialised
to JSON for audit / replay purposes.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionEvent:
    """A timestamped event recorded inside a session."""
    event_type: str
    data:       Any
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "data":       str(self.data),
            "timestamp":  self.timestamp,
        }


class Session:
    """Tracks the lifecycle and artefacts of a HoloLang execution session.

    Parameters
    ----------
    name:
        Human-readable session name.
    """

    def __init__(self, name: str) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.name:       str = name
        self.created_at: float = time.time()
        self.ended_at:   float | None = None
        self._events:    list[SessionEvent] = []
        self._artefacts: dict[str, Any] = {}
        self._skills:    set[str] = set()
        self.metadata:   dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log(self, event_type: str, data: Any = None) -> None:
        self._events.append(SessionEvent(event_type=event_type, data=data))

    def get_events(self, event_type: str | None = None) -> list[SessionEvent]:
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.event_type == event_type]

    # ------------------------------------------------------------------
    # Artefacts
    # ------------------------------------------------------------------

    def store_artefact(self, name: str, value: Any) -> None:
        self._artefacts[name] = value
        self.log("artefact", name)

    def get_artefact(self, name: str) -> Any:
        return self._artefacts.get(name)

    @property
    def artefact_names(self) -> list[str]:
        return list(self._artefacts.keys())

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def add_skill(self, skill_name: str) -> None:
        self._skills.add(skill_name)
        self.log("skill_acquired", skill_name)

    def has_skill(self, skill_name: str) -> bool:
        return skill_name in self._skills

    @property
    def skills(self) -> list[str]:
        return sorted(self._skills)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def end(self) -> None:
        self.ended_at = time.time()
        self.log("session_end")

    @property
    def duration_s(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.created_at

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name":       self.name,
            "created_at": self.created_at,
            "ended_at":   self.ended_at,
            "duration_s": self.duration_s,
            "events":     [e.to_dict() for e in self._events],
            "artefacts":  list(self._artefacts.keys()),
            "skills":     self.skills,
            "metadata":   self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        sess = cls(name=d["name"])
        sess.session_id = d.get("session_id", sess.session_id)
        sess.created_at = d.get("created_at", sess.created_at)
        sess.ended_at   = d.get("ended_at")
        sess._skills    = set(d.get("skills", []))
        sess.metadata   = d.get("metadata", {})
        for ev in d.get("events", []):
            sess._events.append(
                SessionEvent(
                    event_type=ev["event_type"],
                    data=ev["data"],
                    timestamp=ev.get("timestamp", 0.0),
                )
            )
        return sess

    def __repr__(self) -> str:
        return (
            f"Session(name={self.name!r}, id={self.session_id[:8]}, "
            f"events={len(self._events)}, skills={len(self._skills)})"
        )
