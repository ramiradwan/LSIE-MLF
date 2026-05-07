# Operator Console UI/UX Audit

## 1. Purpose, audience, and scope

This artifact describes the **current operator-facing desktop UX/UI system as it exists in code today**.

**Audience.** UX/UI specialist agents preparing future Operator Console design-system work.

**In scope.**
- The PySide6 Operator Console shell and its six operator-facing pages.
- Shared UI primitives, visual system, responsive behavior, and operator-language patterns.
- Adjacent setup/runtime surfaces that directly shape the operator experience.

**Out of scope.**
- Redesign proposals.
- Migration guidance.
- Speculative future design-system structure.
- Manually launched specialist/operator tools outside the agreed surface.
- Deferred or non-wired UI surfaces presented as if they already exist.

---

## 2. Operator experience at a glance

The current desktop operator experience is a **dark, data-dense, trust-oriented Qt shell** built around six persistent routes and one persistent write surface.

- **Overview** is the glance surface.
- **Live Session** is the primary action and trust surface.
- **Experiments**, **Physiology**, **Health**, and **Sessions** are drill-down and management surfaces.
- A single **ActionBar** sits below the content area and stays mounted across route changes.
- Operator-facing language is centralized through formatter helpers rather than built inline in views.
- Responsive behavior is handled by reusable width-band helpers plus page-local column policies.
- The adjacent **Setup** window shares the same dark-neutral family visually, but it is still styled separately from the main console.

---

## 3. Shell architecture and navigation IA

### 3.1 Information architecture

Sidebar order is fixed in `services/operator_console/views/main_window.py`:

1. Overview
2. Live Session
3. Experiments
4. Physiology
5. Health
6. Sessions

Important shell facts:
- **Initial route is Live Session**, even though Overview appears first in the nav.
- All six pages are **eager-instantiated** and mounted into a single `QStackedWidget`.
- Route changes switch stacked pages and invoke optional page `on_activated()` / `on_deactivated()` hooks.
- The shell does not talk to the API directly; it pushes route and selection state into `OperatorStore` and uses `PollingCoordinator` for write actions.

### 3.2 Shell composition

The main shell layout is:
- left sidebar navigation
- right stacked content surface
- persistent ActionBar mounted below the stacked content
- status bar with API/environment readback

Current shell readback behaviors:
- Window title includes the environment label.
- Status bar shows `API · <base_url> · env <label>`.
- Sidebar width scales with window width but is clamped.
- ActionBar switches to compact mode below 1024 px window width.

### 3.3 State model shaping the UX

`services/operator_console/state.py` defines the UI shell vocabulary:
- `AppRoute` is the route enum.
- `StimulusUiContext` models the ActionBar lifecycle state.
- `OperatorStore` is the app-scoped signal hub for route, selection, page data, alerts, and scoped errors.

Current UX consequence:
- The shell preserves selected session context independently from route changes.
- Error banners can stay page-scoped rather than collapsing into one global error state.
- Stimulus lifecycle visuals are driven from one value object, not scattered widget state.

**Primary source files**
- `services/operator_console/views/main_window.py`
- `services/operator_console/state.py`
- `services/operator_console/app.py`
- `services/operator_console/__init__.py`

---

## 4. Surface audit — Overview

### Role in operator workflow
Overview is the **at-a-glance summary** page. It answers: what is active now, what needs attention, and why the latest result counted.

### Major regions
- page header
- page-level warning banner
- six-card responsive metric grid
- attention queue preview below the cards

### Current cards
- Active Session
- Experiment
- Physiology
- Health
- Latest Encounter
- Attention

### Primary actions
- The **Active Session** card is clickable and routes the operator into Live Session with the same session selected.
- Attention items are preview-only on this page; deeper alert history lives in Health.

### State and feedback patterns
- Missing data uses explicit neutral placeholders rather than blank UI.
- Health card differentiates ok / degraded / recovering / error via shared `UiStatusKind` mapping.
- Physiology card distinguishes fresh / stale / absent.
- Latest Encounter surfaces reward inputs compactly instead of only showing a final score.
- Attention queue is capped to a short preview list.

### Trust/readback signals
- Latest Encounter repeats the actual reward inputs used by the pipeline: strongest response signal, semantic gate result, frame count, and compact diagnostics summary.
- Health card summarizes counts rather than a single opaque state.
- Physiology card explains missing or stale data in plain language.

### Responsive notes
- Uses the shared `ResponsiveMetricGrid` helper.
- Attention queue is a simple scrollable preview panel rather than a table.

### Primary source files
- `services/operator_console/views/overview_view.py`
- `services/operator_console/viewmodels/overview_vm.py`
- `services/operator_console/formatters.py`

---

## 5. Surface audit — Live Session

### Role in operator workflow
Live Session is the **main action, readiness, and trust surface**. It is where the operator:
- verifies the phone/runtime is ready
- starts and ends sessions
- sends stimuli
- watches live telemetry
- inspects why an encounter counted or did not count

### Major regions
- page header with ADB and ML status pills
- session header panel
- page-level error banner
- empty state when no session is selected
- setup gate / setup overlay surface
- readiness panel
- live telemetry panel
- phone preview panel
- response signal metric card
- cause/effect panel
- live timeline
- encounter table
- encounter detail panel
- start/end session dialogs

### Setup gating and readiness
Current Live Session behavior explicitly separates setup and readiness states:
- **gate mode** replaces the main dashboard with an empty-state-style setup message
- **overlay mode** keeps the dashboard mounted but overlays setup/calibration messaging
- readiness copy comes from formatter/viewmodel-driven `ReadinessDisplay`
- safe-submit behavior is distinct from deeper pipeline lifecycle details

The page also keeps these distinctions visible:
- no session selected
- setup/connectivity not ready
- calibrating / preparing live analysis
- ready for stimulus
- measuring active response window

### Session controls
Session controls are surfaced through the session header panel:
- start session
- end session
- active arm readback
- expected greeting readback
- calibration/readiness readback

The page itself does not own backend logic; it forwards actions through the viewmodel/coordinator path.

### Telemetry and response monitoring
Current live-monitoring regions show:
- compact live telemetry summary
- response signal card for the strongest observed response signal
- phone preview status panel that explicitly states raw phone frames are not shown
- timeline view that auto-scrolls to latest

### Encounter trust/readback surface
The lower half of the page is built around explaining **why this observed response counted**.

Current trust surfaces include:
- encounters table
- selected encounter detail pane
- reward explanation text
- acoustic detail section
- semantic/attribution diagnostics section
- per-segment physiology freshness/context readback

This page is where the console is most explicit about:
- semantic gate held reward vs measured-but-not-used response signal
- no usable face frames in the response window
- observational voice details
- observational semantic/attribution diagnostics

### Persistent ActionBar relationship
The ActionBar is not embedded inside Live Session, but Live Session is the surface that gives it most of its meaning.

Current relationship:
- shell-level ActionBar reflects the selected live session context
- ActionBar readiness depends on the selected session plus console-derived safe-submit readiness
- Live Session drives the measuring countdown through `StimulusUiContext`
- authoritative timing still comes from orchestrator-owned timestamps; the UI only renders the remaining time

### State and feedback patterns
- page-level warning banner for scoped failures
- empty state for no selected session
- setup gate vs setup overlay distinction
- status pills for ADB and ML backend state
- countdown only when measuring is active
- latest timeline autoscroll
- first encounter auto-selection when possible

### Responsive notes
- page-local responsive breakpoints: 720 / 1040
- dashboard cards reflow using `ResponsiveMetricGrid`
- encounter table and timeline have page-local column policies
- phone preview switches compact presentation on narrow width
- cause/effect and detail panels also adapt to width bands

### Primary source files
- `services/operator_console/views/live_session_view.py`
- `services/operator_console/viewmodels/live_session_vm.py`
- `services/operator_console/widgets/action_bar.py`
- `services/operator_console/formatters.py`

---

## 6. Surface audit — Experiments

### Role in operator workflow
Experiments is the **strategy readback and management surface**. It helps the operator understand which greeting strategy is active and what recent evidence exists for each arm.

### Major regions
- page header
- page-level warning banner
- management panel
- empty state
- three-card summary grid
- strategy evidence panel
- arms table
- latest update panel

### Primary actions
Current write actions are concentrated in the management panel:
- create experiment
- add arm
- rename arm greeting
- disable arm

These controls expose direct operator input fields and buttons rather than a separate modal flow.

### State and feedback patterns
- Empty state when no experiment detail exists.
- Summary cards surface experiment identity, active strategy, and arm count.
- Strategy evidence panel converts arm evidence into card-style summaries.
- Latest update panel provides compact readback of the most recent change.
- Errors surface through a page-level warning banner.

### Trust/readback signals
The page intentionally keeps the adaptive-strategy model readable:
- active arm id
- positive/miss history (`posterior_alpha` / `posterior_beta`)
- uncertainty / variance when present
- selection count
- strongest recent reward
- confirmation rate readback

The page explicitly does **not** frame semantic confidence as part of the reward formula.

### Responsive notes
- page-local responsive breakpoints: 760 / 1040
- summary cards use a 3/2/1 responsive grid
- arm table hides and resizes columns by width band
- management panel also adapts by width

### Primary source files
- `services/operator_console/views/experiments_view.py`
- `services/operator_console/viewmodels/experiments_vm.py`
- `services/operator_console/formatters.py`

---

## 7. Surface audit — Physiology

### Role in operator workflow
Physiology is the **freshness, availability, and co-modulation interpretation surface** for heart-data-derived signals.

### Major regions
- page header
- page-level warning banner
- empty state
- co-modulation summary panel
- streamer role panel
- operator role panel
- detailed co-modulation panel

### Primary actions
This page is read-only. Its job is explanation and status readback, not control.

### State and feedback patterns
The page makes four physiology states explicit:
- fresh
- stale
- absent
- no variability yet

It also treats co-modulation null cases as **informative outcomes**, not subsystem failures.

### Trust/readback signals
Current trust patterns include:
- per-role RMSSD card
- heart-rate card
- freshness card with source timestamp
- provider card
- co-modulation score
- paired observations count
- coverage/window readback
- null-valid explanation using `null_reason`

This page is intentionally careful about distinguishing:
- no snapshot at all
- snapshot exists but RMSSD is not available yet
- stale snapshot
- legitimate null co-modulation due to insufficient aligned pairs

### Responsive notes
- role panels and co-modulation metrics use shared responsive card grids
- no custom table-heavy breakpoint logic on this page

### Primary source files
- `services/operator_console/views/physiology_view.py`
- `services/operator_console/viewmodels/physiology_vm.py`
- `services/operator_console/formatters.py`

---

## 8. Surface audit — Health

### Role in operator workflow
Health is the **readiness and subsystem-trust surface**. It tells the operator what is healthy, what is degraded, what is recovering, and where action is needed.

### Major regions
- header row with repair action and repair status
- page-level warning banner
- empty state
- four-card rollup grid
- readiness checks / probe matrix
- subsystem detail table
- alert timeline

### Primary actions
- **Repair install** is the main explicit operator action on this page.

### State and feedback patterns
The page keeps four distinct operational states separate:
- ok
- degraded
- recovering
- error

Current UI patterns supporting that distinction:
- rollup cards for counts and overall state
- probe matrix with status pills
- subsystem table with explicit next-action/detail wording
- alert timeline for sequence/history
- repair status readback when repair is requested

### Trust/readback signals
Health is explicit about what kind of problem exists:
- degraded is not treated the same as error
- recovering is not treated the same as degraded
- not configured is not treated the same as failure
- table detail copy explains what is happening and what the operator should do next

### Responsive notes
- page-local responsive breakpoints: 760 / 1120
- summary cards use 4/2/1 grid behavior
- subsystem table and alert timeline use width-band column policies
- panel chrome switches into compact handling on narrow widths

### Primary source files
- `services/operator_console/views/health_view.py`
- `services/operator_console/viewmodels/health_vm.py`
- `services/operator_console/formatters.py`

---

## 9. Surface audit — Sessions

### Role in operator workflow
Sessions is the **recent-session history and drill-down launcher**.

### Major regions
- page header
- page-level warning banner
- empty state
- single sessions table

### Primary actions
- single-click selects a session in store state
- double-click or Enter opens that session in Live Session

### State and feedback patterns
- table hidden when there are no rows
- empty state explains that sessions appear after they are started from Live Session
- page-level warning banner handles scoped fetch errors

### Trust/readback signals
This page is deliberately simple. Its main trust value is predictable navigation:
- explicit accessible description and tooltip on the table
- selection does not auto-navigate
- open gesture is deliberate

### Responsive notes
- table uses stretch header behavior rather than a specialized multi-panel layout

### Primary source files
- `services/operator_console/views/sessions_view.py`
- `services/operator_console/viewmodels/sessions_vm.py`

---

## 10. Shared UI primitives and feedback patterns

### SectionHeader
Used at the top of pages and grouped surfaces.
- title + optional subtitle
- accessible name/description synced from visible copy
- layout-only primitive

### MetricCard
The main summary/readback primitive.
- title
- primary value
- optional secondary explanation
- optional embedded status pill
- optional clickability
- keyboard activation when clickable

Current repeated uses:
- overview summary cards
- physiology metrics
- health rollups
- live-session response/readback cards
- experiments evidence cards

### StatusPill
Shared status affordance.
- colored dot + text label
- driven by `UiStatusKind`
- used across action lifecycle, health/readiness, and summary surfaces

### AlertBanner
Page-level attention component.
- severity-specific object names
- hidden when clear
- glyph + message pattern
- reused for warning/info/critical banners

### EmptyStateWidget
Current empty-state primitive.
- centered title + explanatory message
- used to avoid blank surfaces on pages and gated states

### EventTimelineWidget
Reusable timeline/table wrapper.
- generic model attachment
- width-band-aware column policies
- explicit `scroll_to_latest()` behavior

### ActionBar
Persistent shell-level write surface.
- session context line
- expected response readback
- optional operator note input
- submit button
- lifecycle status pill
- optional countdown
- optional progress/error message

Current button-state mapping:
- idle → Send Stimulus
- submitting → Sending…
- accepted → Accepted
- measuring → Measuring…
- completed → Send Stimulus
- failed → Retry

### Repeated feedback patterns across the console
- page-level warning banner for scoped failures
- explicit neutral placeholders instead of silent blanks
- status shown with both wording and color
- detailed explanations routed through formatter helpers
- responsive card-grid reuse instead of per-page ad hoc layout logic

**Primary source files**
- `services/operator_console/widgets/action_bar.py`
- `services/operator_console/widgets/metric_card.py`
- `services/operator_console/widgets/status_pill.py`
- `services/operator_console/widgets/alert_banner.py`
- `services/operator_console/widgets/empty_state.py`
- `services/operator_console/widgets/event_timeline.py`
- `services/operator_console/widgets/section_header.py`

---

## 11. Visual system and theming

### 11.1 Main console visual system
The current Operator Console uses a **dark-neutral palette** defined in `services/operator_console/theme.py`.

Core palette roles:
- app background
- surface
- raised surface
- border
- primary text
- muted text
- accent
- ok / warn / bad status colors
- recovering / degraded health colors

### 11.2 Styling strategy
Current styling is primarily based on:
- one composed application stylesheet from `build_stylesheet()`
- stable widget `objectName` selectors
- shared surface families such as `Panel`, `MetricCard`, `ActionBar`, `AlertBanner*`, `EmptyState`, `EventTimelineTable`

The visual system emphasizes:
- dark panels on dark background
- restrained borders instead of heavy shadows
- muted secondary copy
- accent color for selection/focus/highlight
- severity/status coloring for trust signals

### 11.3 Health/readiness semantics in styling
The theme does not only distinguish ok/warn/error.
It also encodes:
- `HealthRowDegraded`
- `HealthRowRecovering`

This supports a UX distinction that appears throughout the console: **recovering** is not the same as **broken**.

### 11.4 Setup surface relationship
`services/desktop_launcher/ui.py` has a separate setup stylesheet, but it is visibly related:
- same dark background family
- similar border and accent treatment
- similar button emphasis model
- more installer/progress oriented than dashboard oriented

Current reality:
- setup styling is adjacent to the console's visual system
- it is **not yet unified** under the same stylesheet or token source

### 11.5 Current notable implementation detail
Most main-console styling is centralized in the stylesheet, but the shell still contains inline styling for the sidebar title/subtitle in `main_window.py`. That means the current visual system is mostly centralized rather than fully centralized.

**Primary source files**
- `services/operator_console/theme.py`
- `services/operator_console/views/main_window.py`
- `services/desktop_launcher/ui.py`

---

## 12. Responsive behavior and layout rules

### 12.1 Shared responsive system
The shared responsive helpers live in `services/operator_console/widgets/responsive_layout.py`.

Current shared defaults:
- width bands: `NARROW`, `MEDIUM`, `WIDE`
- shared breakpoints: 640 / 980
- shared metric-grid columns: 1 / 2 / 3
- reusable table column policies with per-band visibility, resize mode, and width overrides

### 12.2 Shell-level responsiveness
The shell also has its own responsive behavior:
- sidebar width clamps between 160 and 220 px
- sidebar width scales using 18% of window width
- ActionBar switches into compact mode below 1024 px

### 12.3 Page-local responsive rules
Current notable page-specific behavior:
- **Live Session** uses 720 / 1040 breakpoints and adapts dashboard cards, encounter table, timeline, phone preview, cause/effect panel, and detail panel.
- **Experiments** uses 760 / 1040 breakpoints and adapts cards, strategy evidence, table columns, and management panel.
- **Health** uses 760 / 1120 breakpoints and adapts cards, probe matrix, subsystem table, and alert timeline.
- **Overview** and **Physiology** primarily rely on shared responsive card-grid behavior.
- **Sessions** remains comparatively simple and table-led.

### 12.4 Current responsive design character
This is not a mobile-first console with alternate navigation patterns. The current responsive model is better described as:
- preserve the same information architecture
- reflow cards vertically as width shrinks
- hide lower-priority table columns on narrower widths
- collapse some dense panels into more compact forms

**Primary source files**
- `services/operator_console/widgets/responsive_layout.py`
- `services/operator_console/views/live_session_view.py`
- `services/operator_console/views/experiments_view.py`
- `services/operator_console/views/health_view.py`
- `services/operator_console/views/main_window.py`

---

## 13. Operator language, status semantics, and trust copy

### 13.1 Formatter-driven language system
`services/operator_console/formatters.py` is a first-class part of the UX system.

It centralizes:
- session readback language
- readiness wording
- live telemetry wording
- cause/effect explanations
- reward explanations
- physiology freshness language
- co-modulation interpretation
- health wording
- observational diagnostic copy

### 13.2 Trust-language characteristics
Current operator-facing language is:
- short
- explicit
- causal
- non-speculative
- careful about absence vs invalidity vs failure

Examples of the current language model:
- measured but not used
- no usable face frames
- stimulus confirmed or held back
- null-valid co-modulation
- degraded vs recovering vs error
- no heart-rate variability yet

### 13.3 Readback over abstraction
The UI repeatedly explains **what the system observed and how it treated it**, rather than only exposing backend labels.

This is especially visible in:
- Live Session reward explanation copy
- Overview latest encounter summary
- Physiology stale/absent/no-variability distinctions
- Health next-action details
- ActionBar session/greeting context

### 13.4 Status semantics
A large part of the console's consistency comes from shared status vocabulary:
- `UiStatusKind`
- `AlertSeverity`
- `HealthState`
- `StimulusActionState`

The operator sees the same semantic buckets reused across different surfaces rather than a different wording system on every page.

**Primary source files**
- `services/operator_console/formatters.py`
- `services/operator_console/state.py`
- `services/operator_console/widgets/action_bar.py`

---

## 14. Adjacent setup/runtime surfaces

### 14.1 Setup window
`services/desktop_launcher/ui.py` is the adjacent first-run/setup surface.

Current setup UX includes:
- preparing/installing runtime messaging
- progress bar
- rolling setup log
- retry setup action
- launch LSIE-MLF action
- reinstall runtime action

This is part of the operator experience because it shapes first-run and repair/re-entry behavior before the main console appears.

### 14.2 Desktop runtime framing
`services/desktop_app/__main__.py` shows that the Operator Console is one runtime mode inside the broader desktop app.

Current framing details that affect UX interpretation:
- runtime mode is either `operator_console` or `operator_api`
- preflight runs before child process startup
- the production UI shell is `ui_api_shell`, which launches the loopback operator API runtime and the Qt console together

### 14.3 Production shell path
`services/desktop_app/processes/ui_api_shell.py` applies the same operator-console stylesheet and builds the same main window used by the standalone operator-console entrypoint.

Current implication:
- the Operator Console shell is not only a dev/test surface; it is the production desktop UI shell path

### 14.4 Cloud auth/sync UI absence
Per `docs/DEFERRED_INTEGRATIONS.md`, operator-facing interactive cloud sign-in/sync is **not currently wired into the operator shell**.

Current implications for UX/UI specialists:
- do not assume a current operator-facing sign-in flow exists
- do not assume bundle refresh/sync controls exist in the shell
- do not describe cloud sync producers as current console behavior

**Primary source files**
- `services/desktop_launcher/ui.py`
- `services/desktop_app/__main__.py`
- `services/desktop_app/processes/ui_api_shell.py`
- `docs/DEFERRED_INTEGRATIONS.md`

---

## 15. Explicit omissions / not-yet-wired surfaces

A UX/UI specialist should **not** assume the following currently exist as active operator-facing surfaces:

- operator-facing cloud sign-in UI
- operator-facing cloud sync controls
- operator-facing experiment-bundle refresh UI
- deferred integrations listed in `docs/DEFERRED_INTEGRATIONS.md`
- manually launched specialist tools presented as part of the main shell

Also out of scope for this audit:
- future design tokens/components registry structure
- future design-system migration strategy
- speculative shell redesigns
- speculative navigation or component additions

---

## 16. Source map / evidence note

This audit is grounded in the following current-state sources.

### Shell / runtime
- `services/operator_console/views/main_window.py`
- `services/operator_console/state.py`
- `services/operator_console/app.py`
- `services/operator_console/__init__.py`
- `services/desktop_app/__main__.py`
- `services/desktop_app/processes/ui_api_shell.py`

### Pages
- `services/operator_console/views/overview_view.py`
- `services/operator_console/views/live_session_view.py`
- `services/operator_console/views/experiments_view.py`
- `services/operator_console/views/physiology_view.py`
- `services/operator_console/views/health_view.py`
- `services/operator_console/views/sessions_view.py`

### Shared UI primitives
- `services/operator_console/widgets/action_bar.py`
- `services/operator_console/widgets/metric_card.py`
- `services/operator_console/widgets/status_pill.py`
- `services/operator_console/widgets/alert_banner.py`
- `services/operator_console/widgets/empty_state.py`
- `services/operator_console/widgets/event_timeline.py`
- `services/operator_console/widgets/section_header.py`
- `services/operator_console/widgets/responsive_layout.py`

### Language / theming / adjacent surfaces
- `services/operator_console/formatters.py`
- `services/operator_console/theme.py`
- `services/desktop_launcher/ui.py`
- `docs/DEFERRED_INTEGRATIONS.md`

### Regression expectations reviewed while drafting
- `tests/unit/operator_console/test_main_window.py`
- `tests/unit/operator_console/test_theme.py`
- `tests/unit/operator_console/widgets/test_alert_banner.py`
- `tests/unit/operator_console/widgets/test_event_timeline.py`
