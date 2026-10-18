# Phase Space

A small public laboratory of computational research studies at the intersection of search, optimization, learning and self-organization.

Each study is a bounded investigation: a falsifiable question, credible baselines, validated numerical models, matched compute budgets, an interactive viewer for the evidence, and a readable report that includes limitations and null results.

> **Status:** early. No research results yet. The first study, **Gravity Atlas** (diverse Earth–Moon trajectories in the planar circular restricted three-body problem), has finished its literature scan; its planar CR3BP physics core is validated against the NASA/JPL periodic-orbit catalog, and a static viewer renders the validated reference run. Next: free-return feasibility (M3).

## Studies

| Study | Question (short) | Status |
|---|---|---|
| Gravity Atlas | Does a quality-diversity archive of trajectories stay useful under perturbation better than multi-start optimization at equal budget? | Active |
| The Last Library | When does a strictly bounded inherited memory enable cumulative improvement across generations? | Planned |
| Algorithm Forge | When does evolutionary search over short bit-vector programs beat structured search? | Planned |
| Constitution Machine | Do adaptive resource rules survive changes in the governed population? | Conditional |

See [`docs/`](docs/README.md) for vision, roadmap, design and study specifications.

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just verify   # lint, format check, strict typecheck, tests
just format   # apply formatting and safe lint fixes
```

## Website

`web/` holds the static research site: a gallery of studies and the Gravity Atlas viewer, which renders the validated Earth–Moon L1 Lyapunov reference orbit from versioned, checksummed run files. See [`web/README.md`](web/README.md) to run, test, publish a run, and deploy to Cloudflare Pages. It needs Node.js 22.12 or newer.

```bash
just web-install   # install pinned web dependencies
just web-dev       # local dev server
```

`just verify` runs the Python checks and the web checks and build.

## History

This repository started as *PolicyLab*, an AlphaZero/Hex workbench. That prototype is preserved on branch `archive/policylab-hex`.

## License

MIT OR Apache-2.0.
