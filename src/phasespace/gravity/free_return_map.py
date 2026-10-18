"""Dense maps of free-return feasibility over the departure parameters.

Two sample designs (docs/studies/gravity-atlas/m3-feasibility-design.md, section 5):

- `GridDesign` (S1): deterministic 2-D grid over (phase, delta-v) with alpha = 0; no
  randomness. Row order: phase-major, i.e. index = i_phase * n_delta_v + i_delta_v.
- `SobolDesign` (S4): scrambled Sobol points in the 3-D box (phase, delta-v, alpha) from
  `scipy.stats.qmc.Sobol` with an explicit seed; row order is the generator's order.

`evaluate_points` runs one propagation per point in a process pool. Results are returned
in input order (`Executor.map` preserves order), and each evaluation is a deterministic
function of its point, so the table does not depend on the number of workers.

`sample_table` serializes the records as a columnar JSON table (artifact kind
`sample_table`, docs/data-contract.md): column descriptors plus row-major values, with
null for undefined (NaN) values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Final

import numpy as np
from scipy.stats import qmc

from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    FreeReturnRecord,
    Outcome,
    evaluate,
)
from phasespace.gravity.propagation import Termination

type Point = tuple[float, float, float]
"""(phase_deg, delta_v_km_s, impulse_angle_deg)."""

DEFAULT_CHUNKSIZE: Final = 16


@dataclass(frozen=True, slots=True)
class GridDesign:
    """Uniform 2-D grid; defaults are the S1 grid (0.5 deg x 2 m/s, 720 x 126 = 90 720).

    Phases are phase_min + i * phase_step (i < n_phase), so 360 deg is not repeated;
    delta-v values are delta_v_min + j * delta_v_step (j < n_delta_v), inclusive of
    3.25 km/s for the defaults. Values are rounded to 1e-9 to remove float noise.
    """

    phase_min_deg: float = 0.0
    phase_step_deg: float = 0.5
    n_phase: int = 720
    delta_v_min_km_s: float = 3.00
    delta_v_step_km_s: float = 0.002
    n_delta_v: int = 126
    impulse_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_phase < 1 or self.n_delta_v < 1:
            raise ValueError("grid sizes must be >= 1")
        if not (self.phase_step_deg > 0.0 and self.delta_v_step_km_s > 0.0):
            raise ValueError("grid steps must be positive")

    @property
    def size(self) -> int:
        return self.n_phase * self.n_delta_v

    def points(self) -> list[Point]:
        return [
            (
                round(self.phase_min_deg + i * self.phase_step_deg, 9),
                round(self.delta_v_min_km_s + j * self.delta_v_step_km_s, 9),
                self.impulse_angle_deg,
            )
            for i in range(self.n_phase)
            for j in range(self.n_delta_v)
        ]


DV_PROBE: Final = GridDesign(
    phase_min_deg=229.0,
    phase_step_deg=22.0,
    n_phase=2,
    delta_v_min_km_s=3.120,
    delta_v_step_km_s=0.00002,
    n_delta_v=3501,
)
"""Fine delta-v probe: theta = 229 and 251 deg, delta-v 3.120-3.190 km/s at 0.02 m/s.

Resolves the feasible-band widths that the S1 grid (2 m/s) aliases: family (A) at
229 deg and the narrow window of family (B) at 251 deg (7002 propagations)."""


def feasible_intervals(
    records: Sequence[FreeReturnRecord],
) -> dict[float, list[tuple[float, float, int]]]:
    """Contiguous feasible runs per phase, in delta-v order: (first, last, count).

    `records` must come from a phase-major grid (rows of one phase consecutive and sorted
    by delta-v). A run of n points at step d spans at least (n - 1) d and at most (n + 1) d.
    """
    out: dict[float, list[tuple[float, float, int]]] = {}
    current: list[FreeReturnRecord] = []
    phase: float | None = None

    def close() -> None:
        if current and phase is not None:
            out[phase].append((current[0].delta_v_km_s, current[-1].delta_v_km_s, len(current)))
        current.clear()

    for r in records:
        if r.phase_deg != phase:
            close()
            phase = r.phase_deg
            out.setdefault(phase, [])
        if r.feasible:
            current.append(r)
        else:
            close()
    close()
    return out


@dataclass(frozen=True, slots=True)
class SobolDesign:
    """Scrambled Sobol sample of 2**log2_samples points in the 3-D box (S4 defaults)."""

    seed: int
    log2_samples: int = 17
    phase_bounds_deg: tuple[float, float] = (0.0, 360.0)
    delta_v_bounds_km_s: tuple[float, float] = (3.00, 3.25)
    impulse_angle_bounds_deg: tuple[float, float] = (-10.0, 10.0)

    def __post_init__(self) -> None:
        if not (0 <= self.log2_samples <= 24):
            raise ValueError("log2_samples must be in [0, 24]")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def size(self) -> int:
        return 2**self.log2_samples

    def points(self) -> list[Point]:
        sampler = qmc.Sobol(d=3, scramble=True, rng=np.random.default_rng(self.seed))
        unit = sampler.random_base2(m=self.log2_samples)
        lo = np.array(
            [
                self.phase_bounds_deg[0],
                self.delta_v_bounds_km_s[0],
                self.impulse_angle_bounds_deg[0],
            ]
        )
        hi = np.array(
            [
                self.phase_bounds_deg[1],
                self.delta_v_bounds_km_s[1],
                self.impulse_angle_bounds_deg[1],
            ]
        )
        scaled = qmc.scale(unit, lo, hi)
        return [(float(p[0]), float(p[1]), float(p[2])) for p in scaled]


def _evaluate_point(point: Point, definition: FreeReturnDefinition) -> FreeReturnRecord:
    return evaluate(point[0], point[1], point[2], definition)


def evaluate_points(
    points: Sequence[Point],
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    *,
    workers: int = 1,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> list[FreeReturnRecord]:
    """Evaluate every point (one propagation each); results are in input order."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    task = partial(_evaluate_point, definition=definition)
    if workers == 1:
        return [task(p) for p in points]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(task, points, chunksize=chunksize))


@dataclass(frozen=True, slots=True)
class Column:
    """A table column: name, unit (a data-contract MetricUnit; None: categorical text)."""

    name: str
    unit: str | None
    definition: str


COLUMNS: Final[tuple[Column, ...]] = (
    Column("phase_deg", "deg", "departure phase theta, counterclockwise from +x (Earth->Moon)"),
    Column("delta_v_km_s", "km/s", "TLI impulse magnitude"),
    Column("impulse_angle_deg", "deg", "impulse angle alpha from local horizontal (+: outward)"),
    Column("outcome", None, "feasibility verdict: 'feasible' or the failure reason"),
    Column("termination", None, "propagation termination reason"),
    Column("t_final_days", "day", "time of the last propagated state after TLI"),
    Column(
        "perilune_altitude_km",
        "km",
        "altitude of the closest Moon periapsis before the return shell crossing (or before "
        "t_final without a return); null if none",
    ),
    Column("lunar_encounter_time_days", "day", "time of that closest Moon periapsis"),
    Column(
        "lunar_passage_sense",
        "dimensionless",
        "sign of the rotating-frame Moon-relative angular momentum at perilune (0: none)",
    ),
    Column("moon_periapses_in_band", "count", "Moon periapses in the perilune band before return"),
    Column(
        "earth_periapses_after_encounter",
        "count",
        "Earth periapses between the closest lunar approach and the return (or t_final)",
    ),
    Column("return_time_days", "day", "inbound return-shell crossing time; null if none"),
    Column("return_speed_km_s", "km/s", "Earth-relative inertial speed at the return shell"),
    Column(
        "return_flight_path_angle_deg",
        "deg",
        "Earth-relative inertial flight-path angle at the return shell (< 0: descending)",
    ),
    Column(
        "return_vacuum_perigee_altitude_km",
        "km",
        "osculating two-body (Earth only, no atmosphere) perigee altitude at the return-shell "
        "crossing; null without a return",
    ),
    Column(
        "vacuum_perigee_altitude_km",
        "km",
        "targeting value: osculating vacuum perigee altitude at the first Earth approach "
        "after the closest lunar approach (the shell crossing if no Earth periapsis precedes "
        "it, else the first post-lunar Earth periapsis); null if neither",
    ),
    Column("perigee_source", None, "'return_shell', 'earth_periapsis' or 'none'"),
    Column("jacobi_constant", "normalized_cr3bp", "Jacobi constant of the initial state"),
    Column(
        "jacobi_max_abs_drift_steps",
        "normalized_cr3bp",
        "max |C(s_i) - C(s_0)| over the accepted solver steps",
    ),
    Column("nfev", "count", "right-hand-side evaluations of the propagation"),
)


def _cell(value: object) -> Any:
    if isinstance(value, Outcome | Termination):
        return value.value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def record_row(record: FreeReturnRecord) -> list[Any]:
    """Row values in `COLUMNS` order; NaN becomes None (JSON null)."""
    return [_cell(getattr(record, c.name)) for c in COLUMNS]


def sample_table(
    run_id: str, columns: Sequence[Column], rows: Sequence[Sequence[Any]]
) -> dict[str, Any]:
    """Columnar JSON document of artifact kind `sample_table` (rows in column order)."""
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("every row must have one value per column")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "columns": [{"name": c.name, "unit": c.unit, "definition": c.definition} for c in columns],
        "rows": [list(row) for row in rows],
    }


def records_table(run_id: str, records: Sequence[FreeReturnRecord]) -> dict[str, Any]:
    """`sample_table` of free-return records with the `COLUMNS` layout."""
    return sample_table(run_id, COLUMNS, [record_row(r) for r in records])


def record_from_row(row: Sequence[Any]) -> FreeReturnRecord:
    """Inverse of `record_row` (None becomes NaN)."""
    if len(row) != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} values, got {len(row)}")
    values = dict(zip((c.name for c in COLUMNS), row, strict=True))

    def num(name: str) -> float:
        v = values[name]
        return math.nan if v is None else float(v)

    def count(name: str) -> int:
        return int(values[name])

    source = str(values["perigee_source"])
    if source not in ("return_shell", "earth_periapsis", "none"):
        raise ValueError(f"unknown perigee_source {source!r}")
    return FreeReturnRecord(
        phase_deg=num("phase_deg"),
        delta_v_km_s=num("delta_v_km_s"),
        impulse_angle_deg=num("impulse_angle_deg"),
        outcome=Outcome(values["outcome"]),
        termination=Termination(values["termination"]),
        t_final_days=num("t_final_days"),
        perilune_altitude_km=num("perilune_altitude_km"),
        lunar_encounter_time_days=num("lunar_encounter_time_days"),
        lunar_passage_sense=count("lunar_passage_sense"),
        moon_periapses_in_band=count("moon_periapses_in_band"),
        earth_periapses_after_encounter=count("earth_periapses_after_encounter"),
        return_time_days=num("return_time_days"),
        return_speed_km_s=num("return_speed_km_s"),
        return_flight_path_angle_deg=num("return_flight_path_angle_deg"),
        return_vacuum_perigee_altitude_km=num("return_vacuum_perigee_altitude_km"),
        vacuum_perigee_altitude_km=num("vacuum_perigee_altitude_km"),
        perigee_source=source,
        jacobi_constant=num("jacobi_constant"),
        jacobi_max_abs_drift_steps=num("jacobi_max_abs_drift_steps"),
        nfev=count("nfev"),
    )


def dumps_table(table: dict[str, Any]) -> bytes:
    """Compact JSON: one row per line, no NaN."""
    head = {k: v for k, v in table.items() if k != "rows"}
    text = json.dumps(head, indent=2, allow_nan=False)
    rows = ",\n".join(
        "    " + json.dumps(row, separators=(",", ":"), allow_nan=False) for row in table["rows"]
    )
    body = '  "rows": [\n' + rows + "\n  ]" if table["rows"] else '  "rows": []'
    return (text[:-2] + ",\n" + body + "\n}\n").encode("utf-8")


def outcome_counts(records: Sequence[FreeReturnRecord]) -> dict[Outcome, int]:
    """Number of records per outcome (every outcome listed, zeros included)."""
    counts = dict.fromkeys(Outcome, 0)
    for r in records:
        counts[r.outcome] += 1
    return counts
