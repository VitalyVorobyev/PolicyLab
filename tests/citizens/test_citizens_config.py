"""The C2 configuration module agrees with Appendix A of the validation protocol.

Freeze checklist item 2: every constant of Appendix A is rendered from
`phasespace.citizens.config` and compared line by line with the protocol text, so a
change on either side fails this test.
"""

from __future__ import annotations

import re
from pathlib import Path

from phasespace.citizens import config

PROTOCOL = (
    Path(__file__).resolve().parents[2] / "docs/studies/model-citizens/protocol-validation.md"
)


def _appendix_a() -> list[str]:
    text = PROTOCOL.read_text(encoding="utf-8")
    section = text.split("## Appendix A", 1)[1]
    block = re.search(r"```text\n(.*?)```", section, flags=re.S)
    assert block is not None
    return [line for line in block.group(1).splitlines() if line.strip()]


def _g(x: float) -> str:
    return f"{x:g}"


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _range(seeds: tuple[int, ...]) -> str:
    assert seeds == tuple(range(seeds[0], seeds[-1] + 1))
    return f"{seeds[0]}..{seeds[-1]}"


def _render() -> list[str]:
    c = config
    strata = len(c.SPLIT_STRATA)
    b = c.B3_BOUNDS
    tau = c.TOLERANCE
    return [
        f"split_seed={c.SPLIT_SEED} bc_split_seed={c.BC_SPLIT_SEED} "
        f"fit/val={c.FIT_UNITS_PER_STRATUM}/{c.VAL_UNITS_PER_STRATUM} units per stratum "
        f"({strata * c.FIT_UNITS_PER_STRATUM}/{strata * c.VAL_UNITS_PER_STRATUM})",
        f"B0 eps={_g(c.B0_EPS)} | M0 smoothing={_g(c.M0_SMOOTHING)} | bins: "
        f"{c.N_FLOOR_BINS} floor (B1,M0,P0), {c.N_LEVELS} levels (B3,B23)",
        f"B1 enc=({c.B1_ENCODER[0]},{c.B1_ENCODER[1]}) gru={c.B1_GRU} "
        f"proj=({c.B1_PROJECTION[0]},{c.B1_PROJECTION[1]}) {c.B1_ACTIVATION} bins={c.B1_BINS} "
        f"adam lr={_sci(c.B1_LEARNING_RATE)} batch={c.B1_BATCH} eval_every={c.B1_EVAL_EVERY} "
        f"patience={c.B1_PATIENCE} max_updates={c.B1_MAX_UPDATES} seeds={_range(c.B1_SEEDS)}",
        f"B2 params={c.B2_PARAMS} starts={c.B2_STARTS} seeds={_range(c.B2_SEEDS)} | "
        f"B3 params={c.B3_PARAMS} starts={c.B3_STARTS} seeds={_range(c.B3_SEEDS)} | "
        f"B23 params={c.B23_PARAMS} seeds={_range(c.B23_SEEDS)}",
        f"B3 bounds phi,rho,delta in [{_g(b['phi'][0])},{_g(b['phi'][1])}] "
        f"lambda in [{_g(b['lambda'][0])},{_g(b['lambda'][1])}] "
        f"N0 in [{_g(b['n0'][0])},{_g(b['n0'][1])}] "
        f"theta in [{_g(b['theta'][0])},{_g(b['theta'][1])}]; "
        f"B23 kappa in [{_g(c.B23_KAPPA_BOUNDS[0])},{_g(c.B23_KAPPA_BOUNDS[1])}]; "
        f"payoff scale {_g(c.EWA_PAYOFF_SCALE)}",
        f"surrogate fc={c.SURROGATE_FC} gru={c.SURROGATE_GRU} "
        f"lr={_sci(c.SURROGATE_LEARNING_RATE)} batch={c.SURROGATE_BATCH} "
        f"eval_every={c.SURROGATE_EVAL_EVERY} patience={c.SURROGATE_PATIENCE} "
        f"max_updates={c.SURROGATE_MAX_UPDATES} seeds M1 {_range(c.SURROGATE_SEEDS_M1)} "
        f"M2 {_range(c.SURROGATE_SEEDS_M2)} (step 2: width {c.SURROGATE_STEP2_WIDTH}, "
        f"{_range(c.SURROGATE_STEP2_SEEDS_M1)} / {_range(c.SURROGATE_STEP2_SEEDS_M2)})",
        f"surrogate accept: median_abs<={c.SURROGATE_ACCEPT_MEDIAN_ABS:.1f} "
        f"p90_abs<={c.SURROGATE_ACCEPT_P90_ABS:.1f} "
        f"exclusion_agreement>={c.SURROGATE_ACCEPT_EXCLUSION:.2f}",
        f"S={c.SIM_GAMES} games per (model, mechanism); M2 Exp 4: "
        f"{c.M2_EXP4_SETS_PER_HUMAN_SET} sets per human set; "
        f"sim seed={c.SIM_SEED_BASE}+100m+r",
        f"bootstrap B={c.BOOTSTRAP_RESAMPLES} seed={c.BOOTSTRAP_SEED}; tolerance tau: "
        f"surplus {_g(tau['surplus'])}, gini {_g(tau['gini'])}, "
        f"depletion {_g(tau['depletion_round'])}, active {_g(tau['players_active'])}; "
        f"rule: >={c.OUTCOME_METRICS_REQUIRED} of {len(c.METRICS)} metrics and "
        "Delta-ell CI_lo>0",
        f"B4: K={c.B4_TYPES} types, calib {c.B4_CALIBRATION_GAMES} games "
        f"({c.B4_CALIBRATION_GAMES_PER_COMPOSITION} x {c.B4_COMPOSITIONS} compositions), "
        f"grid step {_g(c.B4_GRID_STEP)}, eval {c.B4_EVAL_GAMES} games/(variant, mechanism), "
        f"retries={c.B4_RETRIES}, timeout={c.B4_TIMEOUT_S} s, "
        f"discard limit {round(100 * c.B4_DISCARD_LIMIT)} %, call ceiling {c.B4_CALL_CEILING}, "
        f"seed {c.B4_SEED}",
        f"B4 CLI: model={c.B4_MODEL} effort={c.B4_EFFORT} cli_version={c.B4_CLI_VERSION} "
        f"env allowlist={','.join(c.B4_ENV_ALLOWLIST)}",
        f"B4 infrastructure retries={c.B4_INFRA_RETRIES} backoff="
        f"{','.join(_g(x) for x in c.B4_INFRA_BACKOFF_S)} s (never discards) | eval reductions "
        f"{c.B4_EVAL_GAMES_REDUCED[0]} (any), {c.B4_EVAL_GAMES_REDUCED[1]} (M1/M2) | "
        f"gating >={c.B4_MIN_GATING_GAMES} complete games",
        f"likelihood pmf floor={_sci(c.PMF_FLOOR)} | must beat marginal: mean W1/SD over "
        f"{len(c.METRICS)} metrics < M0-sim's where M0-sim passes (a)",
        f"L-BFGS-B maxiter={c.LBFGSB_MAXITER} maxfun={c.LBFGSB_MAXFUN} "
        f"ftol={_sci(c.LBFGSB_FTOL)} gtol={_sci(c.LBFGSB_GTOL)} (2-point finite differences) | "
        f"B1 early stopping={c.B1_EARLY_STOPPING_SCORE} batching={c.B1_BATCHING}",
        f"W1 human reference: median over {c.W1_SPLITS} half-splits | Delta-ell bootstrap: "
        "decision-weighted ratio, one resample matrix per regime",
        f"compute ceiling non-LLM {_g(c.COMPUTE_CEILING_HOURS)} h",
    ]


def test_appendix_a_matches_the_configuration_module() -> None:
    assert _render() == _appendix_a()


def test_derived_counts_are_consistent() -> None:
    assert len(config.B4_INFRA_BACKOFF_S) == config.B4_INFRA_RETRIES
    assert tuple(5.0 * 2**k for k in range(6)) == config.B4_INFRA_BACKOFF_S
    assert config.B2_STARTS == len(config.B2_SEEDS) == 10
    assert config.B3_STARTS == len(config.B3_SEEDS) == len(config.B23_SEEDS)
    assert len(config.B3_BOUNDS) == config.B3_PARAMS
    assert config.B4_CALIBRATION_GAMES == (
        config.B4_CALIBRATION_GAMES_PER_COMPOSITION * config.B4_COMPOSITIONS
    )
    # 15 compositions of 4 players over 3 types: C(4 + 3 - 1, 3 - 1) = 15.
    assert config.B4_COMPOSITIONS == 15
    assert config.BC_FIT_GAMES + config.BC_VAL_GAMES == 512


def test_simulation_seeds_follow_the_protocol_rule() -> None:
    assert config.sim_seed("B0", "Interpolating") == 50001
    assert config.sim_seed("B23", "Proportional") == 50507
    assert config.sim_seed("M0-sim", "M2 Exp 4") == 50104
    seeds = {config.sim_seed(m, r) for m in config.SIM_MODEL_CODES for r in config.SIM_REGIME_CODES}
    assert len(seeds) == len(config.SIM_MODEL_CODES) * len(config.SIM_REGIME_CODES)


def test_config_hash_is_a_stable_sha256() -> None:
    first = config.config_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert config.config_sha256() == first
    assert "game" not in config.config_dict()  # only the module's own constants
