from __future__ import annotations

from pathlib import Path

from leso_agent_demo.utils import dump_json


def write_metadata_json(path, data):
    dump_json(path, data)


def write_markdown_summary(path, lines):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(path, _shapes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"")


def write_gds(path, _shapes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"")
