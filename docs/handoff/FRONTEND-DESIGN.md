# Frontend Product & Interaction Design

## Product principle

The frontend is the primary way to **understand** a research result. Design for an observant visitor and a critical investigator, not just the experiment author. Make it visually distinctive, modern, calm, and exceptionally clear. **Avoid default ML dashboard aesthetics** (dense panels of undifferentiated charts).

## Navigation / pages in M1

- **Gallery `/`**: short mission statement and curated tiles for Gravity Atlas (active), Last Library, Algorithm Forge, Constitution Machine (planned). Existing Hex link where feasible. Planned pages must not masquerade as live demos.
- **Gravity `/gravity`**: concise question, dominant orbital viewer, controls and physical/data assumptions; subsections for trajectories, robustness and comparisons as data becomes real.
- **Research article `/gravity/report`**: later, M3; a readable experimental narrative with figures that link back to runs. In M1 a clearly labeled forthcoming report is enough, not an invented publication.

Simple deep links. Avoid an empty navigation forest or authentication UI.

## Gravity Atlas page — functional layout

Desktop: study title and short hypothesis; main *2D scientific orbit viewer* occupying most of the page; compact control rail for trajectory selection, playback, frame and units; secondary pane with selected trajectory metadata and validated status; underneath, progressively revealed performance/robustness comparisons.

Mobile: viewer remains primary, controls stack underneath, text collapses gracefully. The simulation must not be clipped or require horizontal page scroll.

### Orbit viewer M1 controls

- Play / pause (actually animates time index), restart, step forward/backward, draggable time slider; keyboard equivalents.
- Select trajectory if more than one; switching resets/synchronizes playback predictably.
- Read exact sampled coordinates and time; show time in units declared by artifact (`normalized` + converted days where provided).
- Label Earth, Moon, rotating barycentric reference frame, orbit/trajectory legend, units and study data status.
- Pan/zoom optional for M1; when supported, reset-camera control and no loss of precision due to CSS transforms.
- Display source/run ID + scientific provenance, not only a decorative curve.
- When shown, reported Δv, minimum lunar radius, return conditions and feasibility must come from validated data. Otherwise omit, never invent values.
- Adaptive rendering for reduced-motion preference; playback stopped by default when reduced motion is requested.

### For M2 / M3

- Trajectory family archive: scatter/heatmap by defined MAP-Elites descriptor axes; select cell → highlight associated orbit and metrics.
- Perturbation robustness: confidence/quantile plot from **precomputed** simulations; no deceptive random client-side perturbations.
- Method comparison: show computation budget, seeds, feasible counts, dispersion and failure modes prominently.
- Export/download a run's manifest, data and protocol; URL to exact run.

## Visual direction

A restrained scientific editorial identity:

- Large, readable typography; clear grid; generous spacing and thin separators.
- Dark navy/ink viewing canvas with accessible neutral UI; a deliberate light theme is welcome but not required in M1.
- One consistent accent for interactive state, **different line colors and patterns** for separate trajectories, and explicit semantic statuses. Do not use color alone as meaning.
- Purposeful animation only; no constant glowing motion, parallax, or decorative particle storms.
- Use SVG for labels, axes and overlays. Canvas or SVG for trajectory geometry depending on actual sample count; avoid introducing 3D before the 2D case is solid.
- Target desktop 1440px and phone ~390px for manual QA, plus browser zoom 200%.

## Viewer coordinate contract

- Data contract uses canonical barycentric **synodic/rotating** frame; Earth and Moon fixed at x = -mu and x = 1 - mu, y = 0.
- Viewer transforms physical coordinates to screen coordinates using a single deterministic transform; preserve aspect ratio and draw scale.
- On-screen physics coordinates remain data-derived. Spline smoothing must not invent physics. Linear interpolation is okay *only as playback interpolation* between sufficiently dense recorded samples; label interpolated frames if necessary.
- Do not present scientifically meaningless decorative orbits as data.

## Trust, accessibility, and diagnostics

- Visible status `Illustrative / Computed (not validated) / Validated`, plus methodological limitations.
- Loading, empty, unknown schema version and corrupt file errors with actionable text.
- Keyboard focus rings, screen-reader descriptions of controls, logical tab order, accessible chart summaries, non-color-only legends.
- Keep high-frequency animation state local to visualization; avoid rerendering the whole page each frame.
- Track React errors; do not silently swallow unparseable manifests.

## Minimal acceptance demo

A first-time user visits the gallery, opens Gravity Atlas, clicks a trajectory, scrubs to lunar approach, pauses, reads exact numerical position/time, opens its source metadata, and understands whether the run is merely illustrative or scientifically validated. This should take no external services besides static file hosting.
