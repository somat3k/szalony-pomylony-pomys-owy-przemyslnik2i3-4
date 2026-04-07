"""Tests for the docs subsystem (session, skills, directory)."""

import pytest
import os
from hololang.docs.session import Session, SessionEvent
from hololang.docs.skills import SkillRegistry, SkillRecord, get_registry
from hololang.docs.directory import DocDirectory, DocEntry


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_session_create():
    s = Session("my_session")
    assert s.name == "my_session"
    assert s.session_id


def test_session_log():
    s = Session("log_test")
    s.log("test_event", data={"x": 1})
    events = s.get_events("test_event")
    assert len(events) == 1
    assert events[0].event_type == "test_event"


def test_session_get_all_events():
    s = Session("all_ev")
    s.log("a")
    s.log("b")
    assert len(s.get_events()) == 2


def test_session_artefact():
    s = Session("art")
    s.store_artefact("tensor_out", [1, 2, 3])
    assert s.get_artefact("tensor_out") == [1, 2, 3]
    assert "tensor_out" in s.artefact_names


def test_session_skills():
    s = Session("skills_test")
    s.add_skill("laser_control")
    s.add_skill("tensor_basics")
    assert s.has_skill("laser_control")
    assert not s.has_skill("unknown_skill")
    assert "laser_control" in s.skills


def test_session_end():
    s = Session("end_test")
    assert s.ended_at is None
    s.end()
    assert s.ended_at is not None
    assert s.duration_s is not None and s.duration_s >= 0


def test_session_to_json():
    s = Session("json_test")
    s.log("created")
    j = s.to_json()
    assert "json_test" in j
    assert "created" in j


def test_session_save_load(tmp_path):
    s = Session("save_load")
    s.log("run", "start")
    s.add_skill("mesh_canvas")
    path = str(tmp_path / "session.json")
    s.save(path)
    s2 = Session.load(path)
    assert s2.name == "save_load"
    assert s2.has_skill("mesh_canvas")
    assert len(s2.get_events()) >= 1


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

def test_skill_registry_defaults():
    reg = SkillRegistry()
    skills = reg.all_skills()
    assert len(skills) > 10  # has many defaults


def test_skill_registry_get():
    reg = SkillRegistry()
    s = reg.get("laser_control")
    assert s is not None
    assert s.category == "device"


def test_skill_registry_register():
    reg = SkillRegistry()
    reg.register("custom_skill", description="A custom skill", category="misc")
    s = reg.get("custom_skill")
    assert s is not None
    assert s.description == "A custom skill"


def test_skill_registry_by_category():
    reg = SkillRegistry()
    device_skills = reg.list_by_category("device")
    assert len(device_skills) >= 3


def test_skill_registry_json():
    reg = SkillRegistry()
    j = reg.to_json()
    assert "laser_control" in j


def test_skill_level_names():
    rec = SkillRecord("x", level=1)
    assert rec.level_name == "beginner"
    rec2 = SkillRecord("y", level=3)
    assert rec2.level_name == "advanced"


def test_global_registry_singleton():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


# ---------------------------------------------------------------------------
# DocDirectory
# ---------------------------------------------------------------------------

def test_doc_directory_add_entry():
    d = DocDirectory("root")
    e = d.add_entry("intro", content="Hello", tags=["welcome"])
    assert e.name == "intro"
    assert e.content == "Hello"
    assert "welcome" in e.tags


def test_doc_directory_get_entry():
    d = DocDirectory("root")
    d.add_entry("config", content="cfg content")
    e = d.get_entry("config")
    assert e is not None
    assert e.content == "cfg content"


def test_doc_directory_remove_entry():
    d = DocDirectory("root")
    d.add_entry("remove_me")
    d.remove_entry("remove_me")
    assert d.get_entry("remove_me") is None


def test_doc_directory_mkdir_cd():
    root = DocDirectory("root")
    sub = root.mkdir("chapter1")
    assert sub.name == "chapter1"
    assert root.cd("chapter1") is sub


def test_doc_directory_cd_missing():
    root = DocDirectory("root")
    with pytest.raises(KeyError):
        root.cd("nonexistent")


def test_doc_directory_get_or_create():
    root = DocDirectory("root")
    sub = root.get_or_create("new_sub")
    assert sub.name == "new_sub"
    sub2 = root.get_or_create("new_sub")
    assert sub is sub2  # same object


def test_doc_directory_path():
    root = DocDirectory("docs")
    sub  = root.mkdir("chapter")
    leaf = sub.mkdir("section")
    assert leaf.path == "docs/chapter/section"


def test_doc_directory_search():
    root = DocDirectory("search_root")
    root.add_entry("laser_intro", content="Laser beam basics", tags=["laser"])
    root.add_entry("mirror_guide", content="Mirror angle control", tags=["mirror"])
    sub = root.mkdir("advanced")
    sub.add_entry("tensor_ops", content="Tensor matrix operations", tags=["tensor"])

    results = root.search("laser")
    assert any(e.name == "laser_intro" for e in results)

    results_deep = root.search("tensor", deep=True)
    assert any(e.name == "tensor_ops" for e in results_deep)


def test_doc_directory_tree():
    root = DocDirectory("tree_root")
    root.add_entry("readme")
    sub = root.mkdir("sub")
    sub.add_entry("detail")
    tree = root.tree()
    assert "tree_root" in tree
    assert "readme" in tree
    assert "detail" in tree


def test_doc_directory_to_json():
    root = DocDirectory("json_dir")
    root.add_entry("doc1", content="content")
    j = root.to_json()
    assert "doc1" in j
    assert "content" in j


def test_doc_directory_save_to_disk(tmp_path):
    root = DocDirectory("saved")
    root.add_entry("readme", content="# README\nHello!")
    root.mkdir("sub").add_entry("note", content="Note text")
    root.save_to_disk(str(tmp_path))
    assert (tmp_path / "saved" / "readme.md").exists()
    assert (tmp_path / "saved" / "sub" / "note.md").exists()
    content = (tmp_path / "saved" / "readme.md").read_text()
    assert "Hello!" in content


def test_doc_entry_update():
    e = DocEntry("test_entry", content="old")
    old_ts = e.updated_at
    import time; time.sleep(0.01)
    e.update("new content")
    assert e.content == "new content"
    assert e.updated_at >= old_ts
