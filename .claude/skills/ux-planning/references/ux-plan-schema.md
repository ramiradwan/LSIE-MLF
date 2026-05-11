# UX plan schema summary

Canonical file: `automation/schemas/ux_plan.py`

Generic spec-work-item packets are validated by `automation/schemas/spec_work_item.py`. UX plans and work-item packets generated for a cycle may be saved under `automation/work-items/active/`; that directory is gitignored and active instances should not be committed.

## Required top-level fields
- `spec_refs`
- `source_artifacts`
- `page_route`
- `shell`
- `viewmodel`
- `regions`
- `target_files`
- `target_symbols`

## Region shape
Each region defines:
- `name`
- `layout`
- optional `columns`
- `components`

## Component instance shape
Each component instance defines:
- `component`
- `object_name`
- optional `props`
- optional `status_kind_source`
- optional `formatters`
- `a11y`
- `states`
- `responsive`

## Validation guards
- `object_name` must be unique across the whole plan
- `target_files` must be unique
- `target_symbols` must be unique
- unknown fields are rejected
