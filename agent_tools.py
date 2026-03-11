"""High-level tools exposed to the orchestration layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis_tools import build_validation_report, call_validation_api, extract_litho_summary
from job_runner import run_simulation_job
from recipe_tools import list_recipe_params, modify_recipe_by_tag


def run_litho_workflow(
    recipe_path: str,
    layout_path: str,
    out_dir: str,
    fail_mode: str = "none",
    validation_endpoint: str = "mock://validate",
) -> dict[str, Any]:
    """Run simulator, summarize output, validate externally, and merge report."""
    job = run_simulation_job(recipe_path, layout_path, out_dir, fail_mode=fail_mode)
    if job["state"] not in {"finished", "invalid"}:
        return {
            "ok": True,
            "job": job,
            "summary": None,
            "validation": None,
            "report": {
                "ok": True,
                "final_verdict": "fail",
                "reasons": ["job_not_finished"],
            },
        }

    summary = extract_litho_summary(job["paths"]["result_npz"])
    validation = call_validation_api(validation_endpoint, payload={"summary": summary, "job": job})
    report = build_validation_report(summary, validation, job_status=job)
    return {"ok": True, "job": job, "summary": summary, "validation": validation, "report": report}


def create_default_output_dir() -> str:
    output = Path("runs") / "default"
    output.mkdir(parents=True, exist_ok=True)
    return str(output)


__all__ = [
    "list_recipe_params",
    "modify_recipe_by_tag",
    "run_litho_workflow",
    "extract_litho_summary",
    "call_validation_api",
]
