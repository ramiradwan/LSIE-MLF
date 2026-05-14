# Operator Console UI/UX Specialist Brief

## Purpose

This brief is for a UX specialist reviewing the **current Operator Console desktop interface** and suggesting improvements that stay strictly within the existing UI surface.

The goal is to gather ideas that improve clarity, scanability, consistency, readability, and operator confidence **without** requiring backend work, new product features, new routes/pages, new data, or cloud/auth wiring.

## What exists today

The current Operator Console is a **dark, data-dense, trust-oriented desktop UI** built in PySide6/Qt.

It uses:
- a persistent left sidebar for navigation
- a stacked content area with six routes
- a persistent shell-level `ActionBar` below the main content
- a status bar for API/environment readback
- repeated card, banner, timeline, empty-state, and status-pill patterns

Current routes:
1. Overview
2. Live Session
3. Experiments
4. Physiology
5. Health
6. Sessions

The initial route is **Live Session**, even though Overview appears first in the sidebar.

## Current UI/UX highlights by surface

### Overview

Overview is the at-a-glance page.

It currently emphasizes:
- six summary cards
- a short attention preview list
- explicit placeholders instead of blank states
- compact trust readback for the latest encounter

Current UX character:
- quick status scan
- card-heavy layout
- preview-oriented rather than deeply interactive

### Live Session

Live Session is the primary action and trust surface.

It currently combines:
- readiness/status pills
- session controls
- setup gating or overlay messaging
- live telemetry summaries
- response/readback cards
- timeline and encounter history
- encounter detail/drill-down explanation

This is the page where the UI works hardest to explain:
- whether the system is ready
- what happened during a stimulus/response cycle
- why an encounter counted or did not count
- what is being observed versus what is actually used

It is the densest and most operationally important screen.

### Experiments

Experiments is the strategy readback and management page.

It currently includes:
- summary cards
- evidence summaries
- an arms table
- a management panel for experiment and arm changes
- a latest-update panel

Current UX character:
- mixed readback + edit surface
- table-and-form layout
- evidence is visible, but the page is still fairly dense

### Physiology

Physiology is a read-only interpretation surface.

It currently emphasizes:
- freshness and availability of physiology signals
- per-role metric cards
- co-modulation summary and explanation
- careful distinction between absent, stale, not-yet-available, and null-valid states

Current UX character:
- explanatory rather than action-oriented
- card-based and status-heavy
- relies on precise wording to build trust

### Health

Health is the readiness and subsystem-trust page.

It currently includes:
- rollup cards
- readiness/probe matrix
- subsystem detail table
- alert timeline
- repair-install action and repair status readback

Current UX character:
- diagnostic and operational
- multiple status categories beyond simple good/bad
- tries to distinguish degraded, recovering, and error states clearly

### Sessions

Sessions is the simplest page.

It currently provides:
- recent-session history in a single table
- deliberate open behavior (selection does not auto-navigate)
- empty-state explanation when no sessions exist yet

Current UX character:
- straightforward, table-led, predictable
- less visually layered than the other surfaces

### Persistent ActionBar

The `ActionBar` stays mounted across route changes and is the main shell-level write surface.

It currently includes:
- session context
- expected response readback
- optional operator note field
- submit button
- lifecycle status pill
- countdown or progress/error messaging when relevant

Current UX character:
- persistent and operational
- tightly tied to Live Session context
- important for confidence, timing, and action clarity

## Current shared interaction and visual patterns

Across the console, the current UI repeatedly uses:
- dark neutral surfaces with restrained borders
- summary cards for primary readback
- status pills with both wording and color
- page-level alert banners for scoped issues
- empty states instead of blank panels
- table/timeline readback surfaces for history and sequence
- formatter-driven explanatory copy rather than raw technical labels
- responsive card/table reflow rather than alternate navigation patterns

The overall experience is consistent in intent: it tries to show **what the system observed, what state it is in, and how the operator should interpret that state**.

## Constraints for recommendations

Please keep every recommendation strictly within the current UI and current data.

### In scope

UI-only ideas such as:
- layout and grouping improvements
- better visual hierarchy
- scanability and density improvements
- clearer wording and microcopy
- clearer status presentation
- consistency improvements across pages
- responsive-behavior improvements within the same surfaces
- readability and accessibility improvements
- trust/readback clarity improvements
- `ActionBar` clarity and confidence cues

### Out of scope

Please do **not** recommend changes that require:
- backend work
- new product features
- new routes or pages
- new data collection
- API/schema/database work
- cloud sign-in or sync wiring
- new runtime capabilities
- changing what the system measures or stores

In other words: assume the existing routes, current data, and current feature set are fixed. The request is only about improving how the current interface communicates and organizes what already exists.

## Specific questions for the UX specialist

1. Where is the current interface visually or cognitively too dense?
2. Which surfaces would benefit most from stronger hierarchy, grouping, or progressive disclosure?
3. Where does wording or microcopy make the current UI harder to scan or trust?
4. Which statuses, readiness cues, or trust signals are currently hard to interpret quickly?
5. Where does the interface feel inconsistent across pages even though the underlying patterns are similar?
6. Which parts of the current `ActionBar` interaction could be clearer or more confidence-building without changing functionality?
7. Which current tables, panels, or card groups could be reorganized for better operator flow without adding new features?
8. Where could the current responsive behavior be improved while keeping the same information architecture?
9. Which accessibility/readability improvements would have the highest impact within the current UI only?
10. What are the best quick wins versus deeper-but-still-UI-only improvements?

## Requested response format

For each recommendation, please use this structure:

- **Affected surface:**
- **Current UI/UX problem:**
- **Recommended UI-only change:**
- **Expected UX benefit:**
- **Why this does not require backend or feature work:**
- **Priority:** `quick win` | `medium` | `later`

If helpful, you can group recommendations by surface:
- Overview
- Live Session
- Experiments
- Physiology
- Health
- Sessions
- ActionBar
- shared cross-surface patterns

## Framing notes

A few current-state realities to keep in mind while reviewing:
- The console is intentionally trust-oriented, so explanatory readback matters as much as visual polish.
- `Live Session` is the most important and most crowded surface.
- `Health` and `Physiology` rely heavily on nuanced status language, not just binary healthy/unhealthy states.
- `Overview` is meant to support quick scanning rather than detailed management.
- `Sessions` is intentionally simple and should probably stay simple.
- The setup/install surface exists separately from the main console and is not yet unified under the same styling system.
- Operator-facing cloud sign-in/sync UI is not currently present and should not be assumed.

## Deliverable intent

Please return a prioritized set of ideas for improving the **current interface only** so the engineering team can later choose which UI-only improvements to implement.