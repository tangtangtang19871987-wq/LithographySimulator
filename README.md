# Lithography Workflow Assistant Demo

This repository now includes a **complete Python demo** for a constrained lithography workflow assistant.

## What was added

- Subprocess-based external simulator launch (`simulate_stub.py`) through deterministic monitoring (`job_runner.py`).
- Text recipe editing that only updates assignment lines immediately after `#@param` tags (`recipe_tools.py`).
- Skill package import from `skills/*/SKILL.md` with YAML front matter parsing and tool binding (`skill_loader.py`).
- Registry-driven tool validation and per-skill tool restriction (`tool_registry.py`).
- NPZ result summary extraction + API validation (with `mock://validate`) (`analysis_tools.py`).
- Thin orchestration layer that exposes only approved tools to LangChain (`app.py`).

---

## Architecture

### 1) Constrained recipe editing

`recipe_tools.py` implements two safe operations:

- `list_recipe_params(recipe_path)`
- `modify_recipe_by_tag(recipe_path, tag, new_value, output_path=None)`

Why constrained?

- Recipes are process-control artifacts. Allowing free-form edits from LLM output is unsafe.
- The editor only modifies the **single assignment line immediately after the matching** `#@param` tag.
- Source recipes are never overwritten; output is always a copied file.

### 2) Deterministic job monitoring

`job_runner.py` launches `simulate_stub.py` with `subprocess.Popen`, then polls at a fixed interval.

State machine:

- `pending`
- `running`
- `finished`
- `failed`
- `invalid`
- `timeout`

Validation checks include:

- process return code
- timeout
- log scanning for error patterns
- output file existence
- output file size > 0
- NPZ readability
- numerical sanity (finite and non-constant)

Why deterministic?

- Polling logic is fixed, explicit, and not delegated to an LLM.
- Validation rules are explicit code checks and reproducible.

### 3) Result analysis + API validation

`analysis_tools.py` reads NPZ results and returns:

- `source`
- `shape`
- `min`, `max`, `mean`, `std`
- `has_nan`, `has_inf`
- `center_cd_px`
- `hotspot_ratio`
- `binary_area_ratio`

`call_validation_api()` supports:

- `mock://validate` deterministic local validation
- HTTP POST JSON validation via `requests`

`build_validation_report()` combines job state + local summary + API verdict into a final report.

### 4) Skill markdown import + tool binding

`skill_loader.py` loads `skills/*/SKILL.md` and parses YAML front matter fields:

- `name`
- `description`
- `triggers`
- `tools`

Then:

- skills are routed by trigger matching
- declared tools are validated against `ToolRegistry`
- only selected-skill tools are exposed to agent mode

### 5) LangChain as a thin orchestration layer

`app.py` registers these tools:

- `list_recipe_params`
- `modify_recipe_by_tag`
- `run_litho_workflow`
- `extract_litho_summary`
- `call_validation_api`

LLM cannot directly rewrite files or control polling internals; it can only call bound tools.

---

## Files

- `app.py`
- `recipe_tools.py`
- `job_runner.py`
- `analysis_tools.py`
- `agent_tools.py`
- `skill_loader.py`
- `tool_registry.py`
- `simulate_stub.py`
- `skills/litho_recipe_edit/SKILL.md`
- `skills/litho_result_validate/SKILL.md`
- `skills/log_triage/SKILL.md`
- `sample_data/base_recipe.rcp`
- `sample_data/sample_layout.gds`

---

## Run instructions

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Deterministic demo mode

```bash
python app.py --mode deterministic-demo
```

This will:

1. list recipe params
2. edit `sigma_out` into a copied recipe
3. launch simulator stub via subprocess
4. monitor and validate outputs
5. summarize NPZ + call `mock://validate`
6. print combined JSON report

### Skill-loaded agent mode

```bash
export OPENAI_API_KEY=...  # required for live LLM tool-calling
python app.py --mode agent --query "Edit sigma and run workflow validation"
```

The app loads all skills from `skills/*/SKILL.md`, chooses the best trigger match, and restricts visible tools to that skill.

---

## Quick sanity checks

```bash
python app.py --mode deterministic-demo
python job_runner.py --recipe sample_data/base_recipe.rcp --layout sample_data/sample_layout.gds --out-dir runs/manual --fail-mode none
python job_runner.py --recipe sample_data/base_recipe.rcp --layout sample_data/sample_layout.gds --out-dir runs/manual_invalid --fail-mode invalid
```

