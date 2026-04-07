"""Skill registry for HoloLang learning progression tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillRecord:
    """A record of a specific skill and its mastery state."""

    name:        str
    description: str  = ""
    category:    str  = "general"
    level:       int  = 0          # 0=unknown, 1=beginner, 2=intermediate, 3=advanced
    acquired_at: float = field(default_factory=time.time)
    metadata:    dict[str, Any] = field(default_factory=dict)

    LEVELS = {0: "unknown", 1: "beginner", 2: "intermediate", 3: "advanced"}

    @property
    def level_name(self) -> str:
        return self.LEVELS.get(self.level, "custom")

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "description": self.description,
            "category":    self.category,
            "level":       self.level,
            "level_name":  self.level_name,
            "acquired_at": self.acquired_at,
            "metadata":    self.metadata,
        }


class SkillRegistry:
    """Global registry of available skills.

    Skills represent competencies in areas such as:
    * Laser beam control
    * Holographic optics
    * Tensor processing
    * Device configuration
    * Light engineering
    * System architecture
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillRecord] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        defaults = [
            ("laser_control",      "Control laser beam parameters",          "device",        1),
            ("mirror_control",     "Configure galvanised mirror angles",      "device",        1),
            ("sensor_reading",     "Read and interpret sensor data",          "device",        1),
            ("tensor_basics",      "Create and manipulate tensors",           "tensor",        1),
            ("tensor_graph",       "Build computation graphs",                "tensor",        2),
            ("safetensor",         "Use SafeTensor with validation",          "tensor",        2),
            ("tensor_pool",        "Pool and batch tensor operations",        "tensor",        3),
            ("mesh_canvas",        "Design mesh canvas layouts",              "mesh",          1),
            ("tile_impulse",       "Wire tile-to-tile impulse cycles",        "mesh",          2),
            ("kernel_programming", "Write kernel instruction programs",       "vm",            2),
            ("block_pipeline",     "Build generative block pipelines",        "vm",            2),
            ("pool_runtime",       "Configure pooled VM runtimes",            "vm",            3),
            ("grpc_channel",       "Set up gRPC communication channels",      "network",       2),
            ("websocket",          "Use WebSocket streams",                   "network",       2),
            ("webhook_dispatch",   "Dispatch and receive webhooks",           "network",       1),
            ("api_design",         "Design REST API endpoints",               "network",       2),
            ("hololang_basics",    "Write basic HoloLang programs",           "language",      1),
            ("hololang_advanced",  "Advanced HoloLang patterns",              "language",      3),
            ("light_engineering",  "Principles of light manipulation",        "physics",       2),
            ("holographic_optics", "Holographic optical system design",       "physics",       3),
            ("session_management", "Manage execution sessions",               "docs",          1),
            ("skill_tracking",     "Track skill progression",                 "docs",          1),
            ("doc_directories",    "Create structured doc directories",       "docs",          1),
            ("system_architecture","Design multi-layer system architectures", "architecture",  3),
        ]
        for name, desc, cat, level in defaults:
            self._skills[name] = SkillRecord(name=name, description=desc,
                                             category=cat, level=level)

    def register(self, name: str, description: str = "",
                 category: str = "general", level: int = 1) -> SkillRecord:
        rec = SkillRecord(name=name, description=description,
                          category=category, level=level)
        self._skills[name] = rec
        return rec

    def get(self, name: str) -> SkillRecord | None:
        return self._skills.get(name)

    def list_by_category(self, category: str) -> list[SkillRecord]:
        return [s for s in self._skills.values() if s.category == category]

    def all_skills(self) -> list[SkillRecord]:
        return list(self._skills.values())

    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self._skills.values()], indent=2)

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={len(self._skills)})"


# Module-level singleton
_global_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry
