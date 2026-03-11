---
name: log_triage
description: Triage simulator logs and determine failure or invalid-output causes.
triggers:
  - log
  - error
  - failed
  - timeout
tools:
  - run_litho_workflow
---

Use this skill when the request is primarily diagnosis of job failures or invalid runs.

Guidance:
- Focus on simulator state transitions.
- Inspect validator fields (`log`, `npz`) and return concise root-cause notes.
