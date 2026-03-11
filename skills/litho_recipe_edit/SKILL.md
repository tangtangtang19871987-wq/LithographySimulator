---
name: litho_recipe_edit
description: Edit text recipes using #@param tags with constrained assignment-line updates.
triggers:
  - recipe
  - #@param
  - sigma
  - edit
tools:
  - list_recipe_params
  - modify_recipe_by_tag
---

Use this skill when the request is to inspect or edit lithography recipe parameters.

Constraints:
- Never modify the source file in place.
- Only change the assignment line immediately following the matching `#@param` tag.
- Return structured JSON for each edit.
