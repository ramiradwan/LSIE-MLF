# Review checklist

## Required
- [ ] No `setStyleSheet()` outside `services/operator_console/design_system/`
- [ ] No hex literals outside `services/operator_console/design_system/tokens.py`
- [ ] Every QSS-facing selector is represented in `design_system.json`
- [ ] Views/widgets do not import `services.operator_console.api_client`
- [ ] Shared status styling still flows through `UiStatusKind`
- [ ] New operator-facing copy is routed through formatter helpers where appropriate
- [ ] Unit tests cover new object names, bindings, or verifier behavior
- [ ] `python scripts/audit/verifiers/design_system.py --paths services/operator_console` passes

## Recommended
- [ ] Accessibility names and descriptions are explicit
- [ ] Responsive behavior is expressed through existing helpers or page-local policies
- [ ] The current-state audit still matches the implemented UI behavior when shared structure changes
