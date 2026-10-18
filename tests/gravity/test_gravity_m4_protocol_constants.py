"""Cross-check: protocol.md appendix A against `phasespace.gravity.m4.config`.

The protocol text and the configuration are two copies of the same constants. This test
parses the appendix's TOML block and compares it, key by key, with values built from the
configuration objects, so that neither can change without the other.
"""

from __future__ import annotations

import re
import tomllib
from typing import Any

from phasespace.gravity.m4.config import (
    CC_STAR_SAFETY_CAP,
    DE_GRID,
    GATES_MODELS,
    MAP_ELITES_GRID,
    MBH_GRID,
    MS_GRID,
    PROTOCOL,
    PROTOCOL_PATH,
    RNG_STREAMS,
    T2_GRID,
    to_jsonable,
)

_BLOCK = re.compile(
    r"<!-- m4-constants: begin -->\s*```toml\n(.*?)```\s*<!-- m4-constants: end -->", re.DOTALL
)


def protocol_constants() -> dict[str, Any]:
    match = _BLOCK.search(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert match is not None, "protocol.md has no m4-constants block"
    return tomllib.loads(match.group(1))


def _section(obj: object, *exclude: str) -> dict[str, Any]:
    out: dict[str, Any] = to_jsonable(obj)
    for key in exclude:
        out.pop(key)
    return out


def config_constants() -> dict[str, Any]:
    """The appendix's structure, built from the configuration."""
    p = PROTOCOL
    cc = p.grids["CC"][0]
    t_fixed = {MAP_ELITES_GRID[0].n_init, MAP_ELITES_GRID[0].init_batch}
    assert all({m.n_init, m.init_batch} == t_fixed for m in MAP_ELITES_GRID)
    assert all(t2.map_elites == m for t2, m in zip(T2_GRID, MAP_ELITES_GRID, strict=True))
    assert len({(t2.emitter_cap, t2.emitter_half_width) for t2 in T2_GRID}) == 1
    assert len({b.local_cap for b in MBH_GRID}) == 1
    de_fixed = {
        (d.strategy, d.mutation, d.recombination, d.updating, d.restart_seed_stride)
        for d in DE_GRID
    }
    assert len(de_fixed) == 1
    de = DE_GRID[0]
    return {
        "version": p.version,
        "run_budget": p.run_budget,
        "checkpoints": list(p.checkpoints),
        "box": _section(p.box),
        "descriptors": _section(p.descriptors),
        "objective": _section(p.objective),
        "gate": _section(p.gate),
        "ground_truth": _section(p.ground_truth, "m3_prefix_run_id", "refinement_half_width"),
        "local_solve": _section(p.local_solve),
        "methods": {
            "ms_max_propagations_per_start": [m.max_propagations_per_start for m in MS_GRID],
            "mbh_delta_n_fail": [[b.delta, b.n_fail] for b in MBH_GRID],
            "mbh_local_cap": MBH_GRID[0].local_cap,
            "de_population": [d.population for d in DE_GRID],
            "de_effective_population": [d.effective_population for d in DE_GRID],
            "de_strategy": de.strategy,
            "de_mutation": list(de.mutation),
            "de_recombination": de.recombination,
            "de_updating": de.updating,
            "de_restart_seed_stride": de.restart_seed_stride,
            "cc_seed_batch": cc.seed_batch,
            "cc_cap_per_target": cc.cap_per_target,
            "cc_r1_weight": cc.r1_weight,
            "cc_star_safety_cap": CC_STAR_SAFETY_CAP,
            "t_sigma_iso_line": [[m.sigma_iso, m.sigma_line] for m in MAP_ELITES_GRID],
            "t_n_init": MAP_ELITES_GRID[0].n_init,
            "t_init_batch": MAP_ELITES_GRID[0].init_batch,
            "t_batch": MAP_ELITES_GRID[0].batch,
            "t_max_stalled_batches": MAP_ELITES_GRID[0].max_stalled_batches,
            "t2_emitter_cap": T2_GRID[0].emitter_cap,
            "t2_emitter_half_width": T2_GRID[0].emitter_half_width,
            "rng_streams": dict(RNG_STREAMS),
        },
        "seeds": _section(p.seeds),
        "errors": {d.name: _section(d, "name") for d in p.errors},
        "correction": _section(p.correction),
        "gates": {
            "normals_seed": p.gates_normals_seed,
            **{g.name: _section(g, "name") for g in p.correction_execution},
        },
        "constraints": _section(p.constraints),
        "statistics": _section(p.statistics),
    }


def test_protocol_appendix_matches_the_configuration() -> None:
    assert protocol_constants() == config_constants()


def test_config_hash_covers_the_cross_checked_sections() -> None:
    hashed = set(to_jsonable(PROTOCOL))
    expected = {
        "version",
        "run_budget",
        "checkpoints",
        "box",
        "descriptors",
        "objective",
        "gate",
        "ground_truth",
        "local_solve",
        "seeds",
        "errors",
        "correction",
        "correction_execution",
        "gates_normals_seed",
        "constraints",
        "statistics",
        "grids",
        "rng_streams",
        "oracle_seeds",
        "cc_star_safety_cap",
    }
    assert hashed == expected
    assert set(GATES_MODELS) == {"gates_primary", "gates_stress"}


def test_constants_quoted_in_the_protocol_text() -> None:
    """Spot checks of values the prose states (not only the appendix)."""
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    stats = PROTOCOL.statistics
    factor = (stats.z_alpha + stats.z_power) ** 2
    assert f"{factor:.6f}" == "13.315201" and "13.315201" in text
    assert [d.effective_population for d in DE_GRID] == [32, 64, 128]
    assert "NP_eff ∈ {32, 64, 128}" in text
    assert 'updating="deferred"' in text
    assert "default_rng((7401, injection seed))" in text
