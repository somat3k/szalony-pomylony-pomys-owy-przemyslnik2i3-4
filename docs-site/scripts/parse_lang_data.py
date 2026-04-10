#!/usr/bin/env python3
"""
parse_lang_data.py — dynamically parses HoloLang source code to produce:

  1.  docs-site/public/lang-data.json        — machine-readable reference data
  2.  docs-site/content/4.api-reference/2.keywords.md  — auto-generated keywords page
  3.  docs-site/content/4.api-reference/3.ast-nodes.md — auto-generated AST nodes page

Run via:
    python docs-site/scripts/parse_lang_data.py

Or via the GitHub Actions "Update Docs Data" workflow.
"""
from __future__ import annotations

import inspect
import json
import sys
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on the import path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hololang.lang.lexer import KEYWORDS  # noqa: E402
from hololang.lang import ast_nodes  # noqa: E402
from hololang.docs.skills import SkillRegistry  # noqa: E402
import hololang  # noqa: E402

DOCS_CONTENT = REPO_ROOT / "docs-site" / "content"
DOCS_PUBLIC  = REPO_ROOT / "docs-site" / "public"
DOCS_PUBLIC.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract keyword list
# ─────────────────────────────────────────────────────────────────────────────

def collect_keywords() -> list[dict]:
    kws = []
    for word, tt in sorted(KEYWORDS.items()):
        kws.append({"keyword": word, "token_type": tt.name})
    return kws


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract AST node definitions via dataclass introspection
# ─────────────────────────────────────────────────────────────────────────────

def collect_ast_nodes() -> list[dict]:
    nodes = []
    for name in dir(ast_nodes):
        obj = getattr(ast_nodes, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ast_nodes.Node)
            and obj is not ast_nodes.Node
        ):
            try:
                flds = [
                    {"name": f.name, "type": str(f.type) if f.type is not None else "Any"}
                    for f in dc_fields(obj)
                    if f.name not in ("line", "col")
                ]
            except TypeError:
                flds = []
            docstring = inspect.getdoc(obj) or ""
            nodes.append({"name": name, "fields": flds, "doc": docstring})
    nodes.sort(key=lambda n: n["name"])
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract registered skills
# ─────────────────────────────────────────────────────────────────────────────

def collect_skills() -> list[dict]:
    registry = SkillRegistry()
    skills = []
    for skill in sorted(registry.all_skills(), key=lambda s: s.name):
        skills.append({
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "level": skill.level_name,
        })
    return skills


# ─────────────────────────────────────────────────────────────────────────────
# 4. Write lang-data.json
# ─────────────────────────────────────────────────────────────────────────────

def write_json(keywords: list, ast_node_list: list, skills: list) -> None:
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": hololang.__version__,
        "keywords": keywords,
        "ast_nodes": ast_node_list,
        "skills": skills,
    }
    out = DOCS_PUBLIC / "lang-data.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"  ✓  wrote {out.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Write keywords.md
# ─────────────────────────────────────────────────────────────────────────────

def write_keywords_md(keywords: list) -> None:
    lines = [
        "---",
        "title: Keywords",
        "description: Auto-generated list of all HoloLang reserved keywords and their token types.",
        "---",
        "",
        "# Keywords",
        "",
        f"> Auto-generated from source on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · HoloLang v{hololang.__version__}",
        "",
        "| Keyword | Token type |",
        "|---|---|",
    ]
    for kw in keywords:
        lines.append(f"| `{kw['keyword']}` | `{kw['token_type']}` |")
    lines.append("")

    out = DOCS_CONTENT / "4.api-reference" / "2.keywords.md"
    out.write_text("\n".join(lines))
    print(f"  ✓  wrote {out.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Write ast-nodes.md
# ─────────────────────────────────────────────────────────────────────────────

def write_ast_nodes_md(nodes: list) -> None:
    lines = [
        "---",
        "title: AST Nodes",
        "description: Auto-generated reference for all HoloLang AST node types.",
        "---",
        "",
        "# AST Nodes",
        "",
        f"> Auto-generated from source on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · HoloLang v{hololang.__version__}",
        "",
        "All nodes inherit from `Node` (which carries `line` and `col` metadata).",
        "",
    ]
    for node in nodes:
        lines.append(f"## `{node['name']}`")
        if node["doc"]:
            lines.append("")
            lines.append(node["doc"])
        if node["fields"]:
            lines.append("")
            lines.append("| Field | Type |")
            lines.append("|---|---|")
            for f in node["fields"]:
                lines.append(f"| `{f['name']}` | `{f['type']}` |")
        lines.append("")

    out = DOCS_CONTENT / "4.api-reference" / "3.ast-nodes.md"
    out.write_text("\n".join(lines))
    print(f"  ✓  wrote {out.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Write skills.md
# ─────────────────────────────────────────────────────────────────────────────

def write_skills_md(skills: list) -> None:
    lines = [
        "---",
        "title: Skill Registry",
        "description: Auto-generated list of all built-in HoloLang skills.",
        "---",
        "",
        "# Skill Registry",
        "",
        f"> Auto-generated from source on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · HoloLang v{hololang.__version__}",
        "",
    ]
    if skills:
        lines += [
            "| Skill key | Category | Level | Description |",
            "|---|---|---|---|",
        ]
        for s in skills:
            lines.append(f"| `{s['name']}` | {s['category']} | {s['level']} | {s['description'] or '—'} |")
    else:
        lines.append("_No skills currently registered._")
    lines.append("")

    out = DOCS_CONTENT / "4.api-reference" / "4.skills.md"
    out.write_text("\n".join(lines))
    print(f"  ✓  wrote {out.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("⚙  Parsing HoloLang language data …")
    keywords  = collect_keywords()
    ast_nodes_list = collect_ast_nodes()
    skills    = collect_skills()

    write_json(keywords, ast_nodes_list, skills)
    write_keywords_md(keywords)
    write_ast_nodes_md(ast_nodes_list)
    write_skills_md(skills)

    print(f"\n✅  Done — {len(keywords)} keywords, {len(ast_nodes_list)} AST nodes, {len(skills)} skills")


if __name__ == "__main__":
    main()
