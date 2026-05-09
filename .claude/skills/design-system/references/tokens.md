# Token source of truth

Canonical sources:
- `services/operator_console/design_system/tokens.py`
- `services/operator_console/design_system/tokens.json`

## Current palette families
- background
- surface
- surface_raised
- border
- text_primary
- text_muted
- text_inverse
- accent
- status_ok
- status_warn
- status_bad
- status_recovering
- status_degraded

## Rules
- Add new palette values in `tokens.py` first.
- Keep `tokens.json` in sync with `token_manifest()`.
- QSS builders and widgets should reference token attributes, never inline hex.
