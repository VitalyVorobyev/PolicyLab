# Phase Space

A small public laboratory of computational research studies at the intersection of search, optimization, learning and self-organization.

Each study is a bounded investigation: a falsifiable question, credible baselines, validated numerical models, matched compute budgets, an interactive viewer for the evidence, and a readable report that includes limitations and null results.

> **Status:** early. No research results yet. The first study, **Gravity Atlas** (diverse Earth–Moon trajectories in the planar circular restricted three-body problem), is in literature review; the physics core comes next.

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

## History

This repository started as *PolicyLab*, an AlphaZero/Hex workbench. That prototype is preserved on branch `archive/policylab-hex`.

## License

MIT OR Apache-2.0.
