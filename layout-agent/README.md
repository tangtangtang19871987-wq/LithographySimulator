# LESO Agent Demo (DUV 7nm Lower-Metal)

This directory contains a demo blueprint implementation for a **spec-driven lower-metal layout generation agent**.

## Highlights

- Spec DSL + Sampling Plan DSL + Report DSL schemas
- Skills-based orchestrator interfaces (compile/plan/generate/check/repair/report/update/export)
- Deterministic geometry backbone using Manhattan rectangle primitives
- Structured event extraction and policy update loop
- CLI for `compile-spec`, `run-batch`, `summarize`, `update-policy`

## Quick start

```bash
cd layout-agent
python -m leso_agent_demo.cli compile-spec configs/default_spec.yaml
python -m leso_agent_demo.cli run-batch configs/default_spec.yaml --out out/batch_001
python -m leso_agent_demo.cli summarize out/batch_001
python -m leso_agent_demo.cli update-policy out/batch_001/batch_policy_report.json > out/plan_002.yaml
```

## Notes

- This is a demo-oriented architecture, not a foundry-certified deck.
- KLayout and gdstk hooks are scaffolded; pure-Python fallback paths are included.
