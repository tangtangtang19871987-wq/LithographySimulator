"""Lithography workflow assistant demo entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_tools import (
    call_validation_api,
    extract_litho_summary,
    list_recipe_params,
    modify_recipe_by_tag,
    run_litho_workflow,
)
from skill_loader import load_skills, route_skill, validate_skill_tools
from tool_registry import ToolRegistry


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("list_recipe_params", "List #@param recipe tags", list_recipe_params)
    registry.register("modify_recipe_by_tag", "Modify tagged recipe value in a copied file", modify_recipe_by_tag)
    registry.register("run_litho_workflow", "Run simulator + deterministic monitoring + validation", run_litho_workflow)
    registry.register("extract_litho_summary", "Extract summary metrics from litho NPZ", extract_litho_summary)
    registry.register("call_validation_api", "Call result validation endpoint", call_validation_api)
    return registry


def deterministic_demo() -> dict[str, Any]:
    base_recipe = Path("sample_data/base_recipe.rcp")
    layout = Path("sample_data/sample_layout.gds")
    runs = Path("runs/deterministic_demo")
    runs.mkdir(parents=True, exist_ok=True)

    params = list_recipe_params(base_recipe)
    edit = modify_recipe_by_tag(base_recipe, tag="sigma_out", new_value="0.95", output_path=runs / "recipe.rcp")
    workflow = run_litho_workflow(
        recipe_path=edit["output_recipe"],
        layout_path=str(layout),
        out_dir=str(runs),
        fail_mode="none",
        validation_endpoint="mock://validate",
    )
    return {"ok": True, "mode": "deterministic_demo", "params": params, "edit": edit, "workflow": workflow}


def _run_langchain_agent(query: str, allowed_tools: dict[str, Any]) -> dict[str, Any]:
    """Thin LangChain orchestration: skill-selected tools only."""
    try:
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except Exception as exc:
        return {
            "ok": False,
            "error": "LangChain stack unavailable",
            "detail": str(exc),
            "allowed_tools": sorted(allowed_tools.keys()),
        }

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "ok": False,
            "error": "OPENAI_API_KEY not set",
            "allowed_tools": sorted(allowed_tools.keys()),
            "hint": "Set OPENAI_API_KEY to run agent mode, or use --mode deterministic-demo.",
        }

    tool_objs = []
    for name, spec in allowed_tools.items():
        tool_objs.append(StructuredTool.from_function(name=name, description=spec.description, func=spec.fn))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, tool_objs)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return {"ok": True, "agent_response": response, "allowed_tools": sorted(allowed_tools.keys())}


def skill_loaded_agent_mode(query: str) -> dict[str, Any]:
    registry = _build_registry()
    skills = load_skills("skills")
    validation = validate_skill_tools(skills, registry.names())
    selected = route_skill(query, skills)

    if selected is None:
        return {
            "ok": False,
            "error": "No matching skill",
            "available_skills": [s.name for s in skills],
            "tool_validation": validation,
        }

    bound = registry.bind(selected.tools)
    agent_result = _run_langchain_agent(query, bound)

    return {
        "ok": True,
        "mode": "skill_loaded_agent",
        "selected_skill": {
            "name": selected.name,
            "description": selected.description,
            "triggers": selected.triggers,
            "tools": selected.tools,
            "path": str(selected.path),
        },
        "tool_validation": validation,
        "agent": agent_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deterministic-demo", "agent"], default="deterministic-demo")
    parser.add_argument("--query", default="Please edit sigma and run workflow, then validate results.")
    args = parser.parse_args()

    if args.mode == "deterministic-demo":
        result = deterministic_demo()
    else:
        result = skill_loaded_agent_mode(args.query)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
