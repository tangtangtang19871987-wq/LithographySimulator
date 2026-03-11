---
name: litho_result_validate
description: Run lithography jobs and validate NPZ results with local + API checks.
triggers:
  - validate
  - result
  - npz
  - workflow
tools:
  - run_litho_workflow
  - extract_litho_summary
  - call_validation_api
---

Use this skill to execute and verify workflow outputs.

Checklist:
- Run deterministic job monitor.
- Extract summary metrics from NPZ.
- Call external validator (or `mock://validate`).
- Produce a final combined report.
