"""Recipe parsing and constrained editing utilities for lithography demos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PARAM_TAG_RE = re.compile(r"^\s*#@param\s+(?P<tag>[A-Za-z0-9_\-\.]+)(?P<meta>.*)$")
ASSIGN_RE = re.compile(r"^(?P<prefix>\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)(?P<value>[^;]+)(?P<suffix>;.*)$")


def _parse_meta(meta_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in meta_text.strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def list_recipe_params(recipe_path: str | Path) -> dict[str, Any]:
    """List tagged parameters in a recipe file as structured JSON-like data."""
    path = Path(recipe_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    params: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        tag_match = PARAM_TAG_RE.match(line)
        if not tag_match:
            continue
        assignment_idx = None
        assignment_line = None
        for j in range(i + 1, len(lines)):
            if lines[j].strip().startswith("#"):
                continue
            if ASSIGN_RE.match(lines[j]):
                assignment_idx = j
                assignment_line = lines[j]
            break
        if assignment_idx is None or assignment_line is None:
            continue
        assign_match = ASSIGN_RE.match(assignment_line)
        assert assign_match is not None
        params.append(
            {
                "tag": tag_match.group("tag"),
                "metadata": _parse_meta(tag_match.group("meta")),
                "line_tag": i + 1,
                "line_assignment": assignment_idx + 1,
                "current_value": assign_match.group("value").strip(),
            }
        )

    return {"ok": True, "recipe_path": str(path), "param_count": len(params), "params": params}


def modify_recipe_by_tag(
    recipe_path: str | Path,
    tag: str,
    new_value: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Modify only the assignment line immediately following a matching #@param tag.

    The original file is never overwritten.
    """
    src = Path(recipe_path)
    lines = src.read_text(encoding="utf-8").splitlines()

    target_i = None
    for i, line in enumerate(lines):
        tag_match = PARAM_TAG_RE.match(line)
        if tag_match and tag_match.group("tag") == tag:
            target_i = i
            break

    if target_i is None:
        return {"ok": False, "error": f"tag '{tag}' not found", "recipe_path": str(src)}

    assignment_i = None
    for j in range(target_i + 1, len(lines)):
        if lines[j].strip().startswith("#"):
            continue
        if ASSIGN_RE.match(lines[j]):
            assignment_i = j
        break

    if assignment_i is None:
        return {
            "ok": False,
            "error": f"no assignment line after tag '{tag}'",
            "recipe_path": str(src),
            "line_tag": target_i + 1,
        }

    match = ASSIGN_RE.match(lines[assignment_i])
    assert match is not None
    old_value = match.group("value").strip()
    lines[assignment_i] = f"{match.group('prefix')}{new_value}{match.group('suffix')}"

    if output_path is None:
        output = src.with_name(f"{src.stem}.edited_{tag}{src.suffix}")
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "source_recipe": str(src),
        "output_recipe": str(output),
        "tag": tag,
        "line_tag": target_i + 1,
        "line_assignment": assignment_i + 1,
        "old_value": old_value,
        "new_value": new_value,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("--tag")
    parser.add_argument("--new-value")
    args = parser.parse_args()

    if args.tag and args.new_value:
        print(json.dumps(modify_recipe_by_tag(args.recipe, args.tag, args.new_value), indent=2))
    else:
        print(json.dumps(list_recipe_params(args.recipe), indent=2))
