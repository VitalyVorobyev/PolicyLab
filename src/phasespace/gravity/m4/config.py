"""Declared constants of the M4 protocol (docs/studies/gravity-atlas/protocol.md, v1).

Every constant below cites the protocol section it comes from. Method hyperparameters
are the protocol's tuning grids (section 6); which grid entry each method uses in the
pilot and the main runs is *to be set on the tuning set per protocol section 6 ("How
hyperparameters are set")* and is never chosen here. The protocol's appendix A repeats
the key constants in machine-readable form; a test checks that both agree.

`require_frozen` guards every command that would produce comparative evidence: it
refuses unless the protocol's status line says `frozen` and the caller confirmed it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

PROTOCOL_VERSION: Final = "v1"
PROTOCOL_RELATIVE_PATH: Final = "docs/studies/gravity-atlas/protocol.md"
REPO_ROOT: Final = Path(__file__).resolve().parents[4]
PROTOCOL_PATH: Final = REPO_ROOT / PROTOCOL_RELATIVE_PATH

type Unit3 = tuple[float, float, float]
"""A decision vector in the unit cube u in [0, 1]^3 (protocol section 4)."""

# --- section 4: parameterization ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchBox:
    """x = (theta deg, delta-v km/s, alpha deg), affine in u: x = u * (hi - lo) + lo."""

    lo: tuple[float, float, float] = (0.0, 3.00, -10.0)
    hi: tuple[float, float, float] = (360.0, 3.25, 10.0)


# --- section 5: feasibility gate, objective, descriptors ---------------------------------


@dataclass(frozen=True, slots=True)
class Descriptors:
    """d1 = log10(h_p / km) on [2, log10 30 000], d2 = t_ret (days) on [5, 10] (5.5)."""

    d1_bounds: tuple[float, float] = (2.0, math.log10(30_000.0))
    d2_bounds: tuple[float, float] = (5.0, 10.0)
    bins: int = 10
    """Primary grid (10 x 10); `secondary_bins` is the pre-declared sensitivity grid."""
    secondary_bins: int = 20


@dataclass(frozen=True, slots=True)
class Objective:
    """e(x) = |gamma_S(x) - gamma*| in degrees (5.4) and the non-QD merit (section 6)."""

    gamma_target_deg: float = -6.5
    vacuum_perigee_target_km: float = 38.0
    vacuum_perigee_scale_km: float = 10.0
    perilune_log_band: tuple[float, float] = (2.0, 4.4771)
    """r2 uses the protocol's rounded upper edge 4.4771 (log10 30 000 = 4.47712...)."""
    perilune_log_scale: float = 0.1
    undefined_residual: float = 1.0e3
    """Residual component where r is undefined, as in M3 `free_return_search`."""
    infeasible_merit_offset: float = 100.0
    infeasible_merit_cap: float = 1000.0


@dataclass(frozen=True, slots=True)
class NumericalGate:
    """Scaled Jacobi gate (5.2) and elite re-check thresholds (5.6)."""

    jacobi_drift_rel_max: float = 1e-11
    recheck_rtol: float = 1e-13
    recheck_perilune_km: float = 0.010
    recheck_return_time_s: float = 0.1
    recheck_gamma_deg: float = 1e-3
    recheck_max_replacements: int = 3
    """5.6: after a failing elite, at most 3 replacements per cell (<= 4 re-checks)."""
    lsoda_fraction: float = 0.05
    lsoda_seed: int = 7301
    stop_fraction: float = 0.01
    """K2: stop if more than 1 % of elites fail the re-check or of propagations the gate."""


# --- section 5.3: coverage ground truth --------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruth:
    log2_map: int = 20
    map_seed: int = 20261018
    m3_prefix_log2: int = 17
    m3_prefix_run_id: str = "gravity-free-return-map-3d-sobol17-v1"
    saturation_prefix_log2: int = 19
    refinement_seeds: tuple[int, int, int] = (20261019, 20261020, 20261021)
    refinement_points: int = 32
    refinement_half_width: float = 2.0 ** (-20.0 / 3.0)
    """Level-1 half-width in u (the GT map's mean spacing); level L uses / 2^(L-1).
    5.3: one Sobol generator per box, seeded with the level's seed, so every box of a
    level has the same relative pattern (independent of the order of the boxes)."""
    saturation_tolerance: float = 0.10


# --- section 6: methods ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalSolve:
    """`least_squares` settings shared by MS, MBH, CC and the T2 emitter (section 6)."""

    method: Literal["trf"] = "trf"
    jac: Literal["2-point"] = "2-point"
    x_scale: float = 1.0
    tolerance: float = 1e-10
    """ftol = xtol = gtol."""


@dataclass(frozen=True, slots=True)
class MultiStartConfig:
    """MS: c_MS propagations per local solve (section 6: the start's own propagation is
    counted in N but not against c_MS)."""

    max_propagations_per_start: int = 100


@dataclass(frozen=True, slots=True)
class BasinHoppingConfig:
    """MBH: perturbation half-width delta (in u), n_fail, local-solve cap 200."""

    delta: float = 0.005
    n_fail: int = 10
    local_cap: int = 200
    """Includes the hop's own first propagation (it is the solve's initial point)."""


@dataclass(frozen=True, slots=True)
class DifferentialEvolutionConfig:
    """DE: nominal population NP; scipy popsize = NP // 3 (section 6).

    With init="sobol" SciPy rounds the population up to the next power of two, so the
    effective population is 32, 64 or 128 (`effective_population`); maxiter uses it.
    """

    population: int = 30
    strategy: Literal["rand1bin"] = "rand1bin"
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.9
    restart_seed_stride: int = 1000
    updating: Literal["deferred"] = "deferred"
    """Section 6: updating='deferred' makes the result independent of the worker count."""

    @property
    def effective_population(self) -> int:
        """The population SciPy uses with init="sobol": 3 * (NP // 3) rounded up to 2^k."""
        return 2 ** math.ceil(math.log2(3 * (self.population // 3)))


@dataclass(frozen=True, slots=True)
class ContinuationConfig:
    """CC: declared, not tuned (seeding batch 2^10, corrector cap 60, weight 10)."""

    seed_batch: int = 1024
    cap_per_target: int = 60
    r1_weight: float = 10.0


@dataclass(frozen=True, slots=True)
class MapElitesConfig:
    """T: iso+line-dd variation (sigma_iso, sigma_line) in u, n_init fixed at 2^12."""

    sigma_iso: float = 0.002
    sigma_line: float = 0.2
    n_init: int = 4096
    init_batch: int = 1024
    batch: int = 64
    max_stalled_batches: int = 100
    """Section 6: safety stop after 100 consecutive batches without a new propagation."""


@dataclass(frozen=True, slots=True)
class CorridorEmitterConfig:
    """T2: T's loop plus one r1-only least-squares solve per batch (cap 40, box +-0.02).

    Section 6: each batch is the emitter solve plus batch - 1 = 63 variation offspring,
    whose parents are drawn after the emitter's evaluations entered the archive.
    """

    map_elites: MapElitesConfig = field(default_factory=MapElitesConfig)
    emitter_cap: int = 40
    emitter_half_width: float = 0.02


MS_GRID: Final = (MultiStartConfig(100), MultiStartConfig(300))
MBH_GRID: Final = (
    BasinHoppingConfig(0.005, 10),
    BasinHoppingConfig(0.02, 10),
    BasinHoppingConfig(0.02, 30),
)
DE_GRID: Final = (
    DifferentialEvolutionConfig(30),
    DifferentialEvolutionConfig(60),
    DifferentialEvolutionConfig(120),
)
MAP_ELITES_GRID: Final = (
    MapElitesConfig(0.002, 0.2),
    MapElitesConfig(0.005, 0.2),
    MapElitesConfig(0.01, 0.2),
)
T2_GRID: Final = tuple(CorridorEmitterConfig(m) for m in MAP_ELITES_GRID)
"""Section 6: T2 is tuned over T's three (sigma_iso, sigma_line) entries."""
CONTINUATION: Final = ContinuationConfig()

METHODS: Final = ("D", "MS", "MBH", "DE", "CC", "T", "T2")
"""Budget-matched arms (CC* is the unmatched reference, run separately)."""

type MethodConfig = (
    MultiStartConfig
    | BasinHoppingConfig
    | DifferentialEvolutionConfig
    | ContinuationConfig
    | MapElitesConfig
    | CorridorEmitterConfig
    | None
)


def method_grid(method: str) -> tuple[MethodConfig, ...]:
    """The tuning grid of a method (D has none: one `None` entry; CC is fixed)."""
    grids: dict[str, tuple[MethodConfig, ...]] = {
        "D": (None,),
        "MS": MS_GRID,
        "MBH": MBH_GRID,
        "DE": DE_GRID,
        "CC": (CONTINUATION,),
        "T": MAP_ELITES_GRID,
        "T2": T2_GRID,
    }
    if method not in grids:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    return grids[method]


RNG_STREAMS: Final = {"MBH": 1, "T": 2, "T2": 3}
"""Section 6: method-internal randomness (MBH hops, MAP-Elites variation and parents, the
T2 emitter's elite) uses numpy.random.default_rng((seed, stream)) with these stream ids,
separate from the Sobol scrambling generator default_rng(seed)."""

# The validated M3 candidates (runs gravity-free-return-c1-v2 / -c2-v2) seeding CC*.
ORACLE_SEEDS: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("c1", 229.14224099316039, 3.1567704135102614, 0.0),
    ("c2", 225.8162592825737, 3.178011128675068, 0.0),
)
CC_STAR_SAFETY_CAP: Final = 50_000
"""CC* is not budget-matched; this cap only guards against a runaway (protocol: <~6 000)."""

# --- section 7: budget -------------------------------------------------------------------

RUN_BUDGET: Final = 2**14
CHECKPOINTS: Final = (2**11, 2**12, 2**13, 2**14)
SMOKE_MAX_BUDGET: Final = 2**8
"""Smoke runs never exceed 2^8 propagations per method."""


def checkpoints_for(budget: int) -> tuple[int, ...]:
    """The 2^11..2^14 checkpoints scaled to a budget (N/8, N/4, N/2, N)."""
    return tuple(max(1, budget // d) for d in (8, 4, 2, 1))


# --- section 8: seeds --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Seeds:
    tuning: tuple[int, ...] = (9101, 9102)
    pilot: tuple[int, ...] = (9201, 9202, 9203)
    main: tuple[int, ...] = tuple(range(1001, 1011))
    main_extension: tuple[int, ...] = tuple(range(1011, 1017))
    bootstrap: int = 4242


SMOKE_SEED: Final = 9001
"""Smoke runs use a seed outside every protocol seed list."""
SMOKE_ERROR_SEED: Final = 9002
"""Error draws of the smoke cost probe (not T_E 7001, not H_E 7101/7102)."""

# --- section 9.1: injection errors and the correction ------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorDistribution:
    name: str
    sigma_magnitude: float
    """Relative TLI magnitude error (0.03 % = 3e-4)."""
    sigma_direction_deg: float
    sigma_timing_s: float
    draws: int
    seed: int
    held_out: bool
    executions: tuple[str, ...] = ("gates_primary",)
    """The MCC execution-error sets evaluated with this injection set (9.1 pairing)."""
    truncation_sigmas: float = 3.0


T_E: Final = ErrorDistribution("T_E", 3e-4, 0.02, 0.1, 32, 7001, held_out=False)
H_E_NOMINAL: Final = ErrorDistribution(
    "H_E_nominal",
    3e-4,
    0.02,
    0.1,
    50,
    7101,
    held_out=True,
    executions=("gates_primary", "gates_stress"),
)
H_E_WIDE: Final = ErrorDistribution("H_E_wide", 6e-4, 0.04, 0.2, 50, 7102, held_out=True)
ERROR_DISTRIBUTIONS: Final = {d.name: d for d in (T_E, H_E_NOMINAL, H_E_WIDE)}


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """One in-plane MCC at t_c, commanded by chord iterations on a fixed Jacobian (9.1).

    Survival (D25): a draw survives at budget B iff it is feasible without correction, or
    the commanded correction has |dv| <= B and the trajectory after the *executed*
    correction (Gates-model errors) is feasible. Chord convergence is a diagnostic.
    """

    t_c_days: float = 1.0
    fd_step_m_s: float = 0.1
    cond_max: float = 1e8
    """2-norm condition number of J; J's rows are both km and its columns both m/s, so the
    value does not depend on the unit choice."""
    max_iterations: int = 3
    tolerance_km: float = 1.0
    """Early stop of the chord iterations (and the convergence diagnostic)."""
    divergence_m_s: float = 100.0
    """The chord stops when |dv| exceeds this; it is above every B in the grid."""
    budget_grid_m_s: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    calibration_band: tuple[float, float] = (0.2, 0.8)


@dataclass(frozen=True, slots=True)
class GatesModel:
    """Execution errors of the commanded MCC (D24; Gates model as used by Burnett, Boone &
    Michelotti, arXiv:2607.03424, 2026, Eq. 81), planar:

    Q = (sigma_r^2 + V^2 sigma_s^2) e e^T + (sigma_a^2 + V^2 sigma_p^2) (I - e e^T),
    V = |dv_m|, e = dv_m / V; I - e e^T is the single in-plane transverse direction.

    The executed burn is dv_m + eta, eta ~ N(0, Q) (not truncated); a zero commanded burn
    is not executed and has no error. Draw k's standard normal pair comes from
    default_rng((GATES_NORMALS_SEED, injection seed)) and is shared by every Gates set,
    every solution and every method (common random numbers, 9.1).
    """

    name: str
    sigma_s: float
    """Proportional magnitude error (dimensionless)."""
    sigma_r_m_s: float
    """Fixed magnitude error along the burn, m/s."""
    sigma_p_rad: float
    """Proportional pointing error, rad."""
    sigma_a_m_s: float
    """Fixed transverse error, m/s (in-plane only in the planar model)."""
    held_out: bool


GATES_PRIMARY: Final = GatesModel("gates_primary", 5e-3, 0.010, 3e-4, 0.0009, False)
GATES_STRESS: Final = GatesModel("gates_stress", 3e-2, 0.010, 3e-4, 0.0009, True)
"""Held-out stress set: sigma_s = 3e-2 (Burnett et al.'s pessimistic value), others primary.
It is evaluated only with H_E-nominal, on the same draws and normals as the primary set."""
GATES_MODELS: Final = {g.name: g for g in (GATES_PRIMARY, GATES_STRESS)}
GATES_NORMALS_SEED: Final = 7401


# --- section 9.2: changed constraints ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstraintFamilies:
    sets_per_family: int = 32
    tuning_per_family: int = 10
    generation_seed: int = 7201
    split_seed: int = 7202
    log_h_lo: tuple[float, float] = (2.0, 4.0)
    width_dex: tuple[float, float] = (0.5, 2.0)
    t_max_days: tuple[float, float] = (6.5, 10.0)
    gamma_c_deg: tuple[float, float] = (-8.0, -5.0)
    corridor_width_deg: tuple[float, float] = (0.5, 2.0)
    h_cap_km: float = 30_000.0
    """9.2: one Sobol generator per family, each seeded with 7201 (dimensions 3, 2, 5 for
    F1, F2, F3), and one default_rng(7202) drawing the three permutations in the order F1,
    F2, F3; the first 10 permuted sets of each family are tuning sets."""


# --- section 8 / 11: statistics ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Statistics:
    alpha: float = 0.05
    holm_family_size: int = 10
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 4242
    confidence: float = 0.95
    effect_p1: float = 2.0
    effect_p2: float = 0.10
    z_alpha: float = 2.807
    z_power: float = 0.842
    seed_count_bounds: tuple[int, int] = (10, 16)
    h2_permutations: int = 10_000
    h2_permutation_seed: int = 4242


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """All declared constants; its canonical JSON is hashed into every M4 manifest."""

    version: str = PROTOCOL_VERSION
    box: SearchBox = field(default_factory=SearchBox)
    descriptors: Descriptors = field(default_factory=Descriptors)
    objective: Objective = field(default_factory=Objective)
    gate: NumericalGate = field(default_factory=NumericalGate)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    local_solve: LocalSolve = field(default_factory=LocalSolve)
    run_budget: int = RUN_BUDGET
    checkpoints: tuple[int, ...] = CHECKPOINTS
    seeds: Seeds = field(default_factory=Seeds)
    errors: tuple[ErrorDistribution, ...] = (T_E, H_E_NOMINAL, H_E_WIDE)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    correction_execution: tuple[GatesModel, ...] = (GATES_PRIMARY, GATES_STRESS)
    gates_normals_seed: int = GATES_NORMALS_SEED
    constraints: ConstraintFamilies = field(default_factory=ConstraintFamilies)
    statistics: Statistics = field(default_factory=Statistics)
    grids: dict[str, tuple[Any, ...]] = field(
        default_factory=lambda: {m: method_grid(m) for m in METHODS}
    )
    rng_streams: dict[str, int] = field(default_factory=lambda: dict(RNG_STREAMS))
    oracle_seeds: tuple[tuple[str, float, float, float], ...] = ORACLE_SEEDS
    cc_star_safety_cap: int = CC_STAR_SAFETY_CAP


PROTOCOL: Final = ProtocolConfig()


def to_jsonable(value: object) -> Any:
    """Dataclasses, tuples and dicts as plain JSON values (floats kept exact)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return {str(k): to_jsonable(v) for k, v in items.items()}
    if isinstance(value, tuple | list):
        seq = cast(Sequence[object], value)
        return [to_jsonable(v) for v in seq]
    return value


def config_json(config: ProtocolConfig = PROTOCOL) -> dict[str, Any]:
    return to_jsonable(config)


def config_sha256(config: ProtocolConfig = PROTOCOL) -> str:
    """sha256 of the canonical (sorted-key, compact) JSON of the configuration."""
    text = json.dumps(config_json(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- freeze guard --------------------------------------------------------------------------

_STATUS_LINE: Final = re.compile(r"^\|\s*Status\s*\|\s*\**`([a-z]+)`")


class ProtocolNotFrozenError(RuntimeError):
    """Raised by every command that would produce comparative evidence before the freeze."""


def protocol_status(path: Path = PROTOCOL_PATH) -> str:
    """The backticked word of the protocol header's `Status` row (e.g. 'draft')."""
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _STATUS_LINE.match(line)
        if match:
            return match.group(1)
    raise ValueError(f"{path}: no '| Status | `...`' row in the header")


def require_frozen(confirm_frozen: bool, path: Path = PROTOCOL_PATH) -> None:
    """Refuse unless the protocol status is `frozen` and the caller confirmed it."""
    status = protocol_status(path)
    if status != "frozen":
        raise ProtocolNotFrozenError(
            f"protocol {path.name} has status {status!r}; comparative runs and held-out "
            "evaluations are allowed only after the freeze (protocol section 1)"
        )
    if not confirm_frozen:
        raise ProtocolNotFrozenError("pass --confirm-frozen to run a frozen-protocol stage")
