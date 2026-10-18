# Phase Space website

The public, static research site: a gallery of studies and the Gravity Atlas orbit viewer. It renders only computed data published under `public/research/` and needs no backend.

Stack: Vite 8, React 19, TypeScript 6 (strict), Tailwind CSS 4, Vitest 5 with Testing Library, ESLint 10 with typescript-eslint and react-hooks. Package manager: **npm**; versions are pinned exactly in `package.json` and locked in `package-lock.json`. Fonts (Source Serif 4, IBM Plex Sans) are self-hosted through Fontsource, so the site makes no third-party requests.

## Run

Requires Node.js 22.12 or newer (developed on Node 24).

```bash
just web-install   # npm ci
just web-dev       # dev server on http://localhost:5173
```

Routes: `/` (gallery), `/gravity` (Gravity Atlas; `/gravity?run=<run_id>` selects a trajectory). Anything else shows the in-app "Page not found".

## Check and build

```bash
just web-verify    # lint, typecheck, tests, production build
just verify        # Python checks and tests, then web-verify
```

or inside `web/`: `npm run lint`, `npm run typecheck`, `npm test`, `npm run build` (output in `dist/`), `npm run preview` to serve the build locally.

## Layout

```text
src/lib/data/        loader: fetch, strict validation (schema v1), SHA-256 verification, typed errors
src/lib/viewer/      the single data -> screen transform, camera bounds, L1 position
src/features/gallery gallery page and study copy
src/features/gravity orbit viewer, playback store and controls, evidence and provenance
public/research/     published runs (immutable) and catalog.json
scripts/             publish-run.mjs
```

### How data is loaded

`public/research/catalog.json` lists studies and their published runs; each entry names the `kind` of the run's single artifact (`trajectory`, `sample_table` or `outcome_grid`), and the loader refuses a manifest whose artifact has another kind. For each run the loader:

1. rejects any path that is not a plain relative path (no `..`, absolute paths, URLs, backslashes or percent-encoding);
2. fetches `manifest.json` and validates it strictly against the schema v1 definition in `docs/data-contract.md`: schema version, every required field and type, every enum (an unknown value such as a new `length_scale_convention` is an error, not a fallback), `null` only where the contract allows it, finite numbers, the redundant fields (`velocity_scale_km_s`, primaries at `(-mu, 0)` and `(1 - mu, 0)`) consistent to 1e-12, each check's `passed` equal to `observed !== null && observed <= threshold`, and a `validated` status only with recorded, passing checks. A check whose `observed` is `null` (a non-finite result) is shown as "non-finite" and failed;
3. fetches the artifact, checks its SHA-256 with WebCrypto against the manifest, and only then parses it:
   - trajectory: run_id match, frame and units, `state_order` exactly `t, x, y, vx, vy`, at least two rows, finite values, strictly increasing time, and optional `events` (snake_case names, times inside the sampled range and in order, four finite state values);
   - sample table: unique snake_case columns with a contract unit or `null`, one value per column per row, finite numbers in numeric columns and strings in text columns (or `null`);
   - outcome grid: axes, `phase_major` order, `base64_uint8` codes decoding to exactly one code per cell and within the outcome table, feasible rows matching the feasible cells in index order, and outcome counts equal to the manifest's `outcome_<name>_count` metrics.

   M3 manifest fields are checked too: `deg` units, `termination` and outcome enums, `solver.return_shell` (with `altitude_km = radius_km − Earth radius`), `derived_from` provenance, and a dirty working tree must be disclosed in `limitations`.

Any failure shows an error explaining what went wrong and how to fix it; nothing from a failing run is drawn.

### Viewer conventions

The viewer draws the barycentric rotating (synodic) frame with one deterministic transform (`fitTransform`): equal x and y scale, y up, centered in the plot. Earth and Moon are drawn at their manifest positions and at true radius when that is at least 2.5 px; smaller bodies are enlarged and marked with a dashed ring. L1 is computed from `mu` and labeled as such. Playback steps through recorded samples only; nothing is interpolated. Animation state lives in a small store outside React, driven by `requestAnimationFrame`, so frames re-render only the marker, slider and readout. Playback starts paused when the visitor prefers reduced motion.

Keyboard, with the viewer focused: Space plays or pauses, Left and Right step one sample (ten with Shift), Home goes to the start, End to the last sample.

## Publish a run

Runs are exported by the Python code into `runs/` (ignored by git), for example with `just gravity-reference`. To publish one:

```bash
just web-publish-run gravity-l1-lyapunov-reference-v2 gravity-atlas/l1-lyapunov-reference-v2
# equivalent: node web/scripts/publish-run.mjs runs/<run_id> <destination under web/public/research>
```

The script verifies every artifact's SHA-256 against the manifest, refuses manifests whose provenance records uncommitted changes (`source.git_dirty` must be `false`), copies the files byte for byte, verifies the copies, and refuses to overwrite an existing directory. For local development only, `--allow-dirty-for-development` accepts a dirty run whose limitations disclose it; the site then shows a "Development build" warning, and the run must be replaced by a clean re-export before release. Published runs are immutable: to change one, export and publish a new run under a new directory. Then add the printed entry to `public/research/catalog.json`; the tests load and verify every run the catalog lists.

## Deploy to Cloudflare Pages

Nothing is deployed automatically. With Git integration, configure the Pages project as:

| Setting | Value |
|---|---|
| Root directory | `web` |
| Build command | `npm ci && npm run build` |
| Build output directory | `dist` |
| Environment variable | `NODE_VERSION=24` (any version from 22.12 works) |

Client-side routes: `public/_redirects` rewrites `/gravity` and `/gravity/*` to `index.html` with status 200. Static files, including everything under `/research/`, match before these rules, so a missing JSON file still returns 404 instead of an HTML page. Other unknown paths fall back to `index.html` through Pages' single-page-application default (the build has no top-level `404.html`), and the app renders its own not-found page. Check both behaviours on the real host after the first deployment.

With Wrangler instead of Git integration: `npm run build`, then `npx wrangler pages deploy dist` from `web/` (requires your Cloudflare account; no credentials are stored in this repository).
