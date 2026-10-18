"""Compact, lossless outcome grid of a 2-D free-return map, for the static website.

The S1 map table (`samples.json`, ~19 MB, one row per departure) is too large to publish.
`outcome_grid` keeps, without downsampling, the one thing the viewer draws for every cell,
the feasibility outcome, plus a few columns of the feasible cells:

- `codes`: one uint8 per grid cell in the map's phase-major order (index = i_phase *
  n_delta_v + i_delta_v); code k means `outcomes[k]`. The bytes are base64-encoded
  (`encoding = "base64_uint8"`), so the cell count (90 720 for S1) costs ~121 kB.
- `feasible`: a `sample_table`-shaped block (columns + rows) with the grid indices and the
  perilune altitude, lunar encounter time, return time and lunar passage sense of every
  feasible cell, copied verbatim from the source rows.

The grid axes are those of the source run's `GridDesign` (from its `grid_*` metrics); each
source row's phase and delta-v must equal the design point at its index, so the encoding
cannot silently reorder or drop cells. The format is the `OutcomeGrid` artifact of
docs/data-contract.md.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from phasespace.gravity.free_return import Outcome
from phasespace.gravity.free_return_map import Column, GridDesign

ARTIFACT_KIND: Final = "outcome_grid"
ENCODING: Final = "base64_uint8"
ORDER: Final = "phase_major"
OUTCOMES: Final[tuple[Outcome, ...]] = tuple(Outcome)
"""Code table: code k <-> OUTCOMES[k] (the declaration order of `Outcome`)."""

_FEASIBLE_SOURCE_COLUMNS: Final = (
    "perilune_altitude_km",
    "lunar_encounter_time_days",
    "return_time_days",
    "lunar_passage_sense",
)
FEASIBLE_COLUMNS: Final[tuple[Column, ...]] = (
    Column("phase_index", "count", "grid phase index i_phase (phase = min + i_phase * step)"),
    Column("delta_v_index", "count", "grid delta-v index i_delta_v (delta-v = min + i * step)"),
    Column(
        "perilune_altitude_km",
        "km",
        "altitude of the closest Moon periapsis before the return-shell crossing (source column)",
    ),
    Column("lunar_encounter_time_days", "day", "time of that Moon periapsis after TLI"),
    Column("return_time_days", "day", "inbound return-shell crossing time after TLI"),
    Column(
        "lunar_passage_sense",
        "dimensionless",
        "sign of the rotating-frame Moon-relative angular momentum at perilune",
    ),
)

_GRID_METRICS: Final = (
    "grid_phase_min_deg",
    "grid_phase_step_deg",
    "grid_phase_count",
    "grid_delta_v_min_km_s",
    "grid_delta_v_step_km_s",
    "grid_delta_v_count",
    "impulse_angle_deg",
)


def design_from_metrics(metrics: Mapping[str, Any]) -> GridDesign:
    """The `GridDesign` recorded in a grid map run's `grid_*` metrics."""
    missing = [k for k in _GRID_METRICS if k not in metrics]
    if missing:
        raise ValueError(f"not a grid map run: missing metrics {missing}")

    def value(key: str) -> float:
        return float(metrics[key]["value"])

    return GridDesign(
        phase_min_deg=value("grid_phase_min_deg"),
        phase_step_deg=value("grid_phase_step_deg"),
        n_phase=int(metrics["grid_phase_count"]["value"]),
        delta_v_min_km_s=value("grid_delta_v_min_km_s"),
        delta_v_step_km_s=value("grid_delta_v_step_km_s"),
        n_delta_v=int(metrics["grid_delta_v_count"]["value"]),
        impulse_angle_deg=value("impulse_angle_deg"),
    )


def encode_codes(codes: bytes) -> str:
    """Base64 (RFC 4648, with padding) of the raw code bytes."""
    return base64.b64encode(codes).decode("ascii")


def decode_codes(text: str, size: int) -> bytes:
    """Inverse of `encode_codes`; raises ValueError unless exactly `size` bytes decode."""
    try:
        data = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError(f"codes are not valid base64: {exc}") from exc
    if len(data) != size:
        raise ValueError(f"expected {size} codes, decoded {len(data)}")
    return data


def outcome_grid(run_id: str, design: GridDesign, table: Mapping[str, Any]) -> dict[str, Any]:
    """`OutcomeGrid` document of a phase-major grid map table (`sample_table` of S1 rows).

    Raises ValueError if the table does not hold exactly the design's points in order, has
    an unknown outcome, or lacks a feasible cell's copied columns.
    """
    names = [str(c["name"]) for c in table["columns"]]
    needed = (
        "phase_deg",
        "delta_v_km_s",
        "impulse_angle_deg",
        "outcome",
        *_FEASIBLE_SOURCE_COLUMNS,
    )
    missing = [n for n in needed if n not in names]
    if missing:
        raise ValueError(f"table lacks columns {missing}")
    col = {n: names.index(n) for n in needed}
    rows: Sequence[Sequence[Any]] = table["rows"]
    points = design.points()
    if len(rows) != len(points):
        raise ValueError(f"table has {len(rows)} rows, the grid has {len(points)} cells")
    code_of = {o.value: k for k, o in enumerate(OUTCOMES)}
    codes = bytearray(len(rows))
    feasible: list[list[Any]] = []
    for k, (row, point) in enumerate(zip(rows, points, strict=True)):
        observed = (row[col["phase_deg"]], row[col["delta_v_km_s"]], row[col["impulse_angle_deg"]])
        if observed != point:
            raise ValueError(f"row {k} is {observed}, expected grid point {point}")
        outcome = row[col["outcome"]]
        if outcome not in code_of:
            raise ValueError(f"row {k}: unknown outcome {outcome!r}")
        codes[k] = code_of[outcome]
        if outcome == Outcome.FEASIBLE.value:
            copied = [row[col[n]] for n in _FEASIBLE_SOURCE_COLUMNS]
            if any(v is None for v in copied):
                raise ValueError(f"row {k}: feasible cell without {_FEASIBLE_SOURCE_COLUMNS}")
            feasible.append([k // design.n_delta_v, k % design.n_delta_v, *copied])
    return {
        "schema_version": 1,
        "run_id": run_id,
        "phase_axis": {
            "name": "phase_deg",
            "unit": "deg",
            "min": design.phase_min_deg,
            "step": design.phase_step_deg,
            "count": design.n_phase,
        },
        "delta_v_axis": {
            "name": "delta_v_km_s",
            "unit": "km/s",
            "min": design.delta_v_min_km_s,
            "step": design.delta_v_step_km_s,
            "count": design.n_delta_v,
        },
        "impulse_angle_deg": design.impulse_angle_deg,
        "order": ORDER,
        "outcomes": [o.value for o in OUTCOMES],
        "encoding": ENCODING,
        "codes": encode_codes(bytes(codes)),
        "feasible": {
            "columns": [
                {"name": c.name, "unit": c.unit, "definition": c.definition}
                for c in FEASIBLE_COLUMNS
            ],
            "rows": feasible,
        },
    }


def decode_outcomes(grid: Mapping[str, Any]) -> list[str]:
    """Outcome name of every cell, in the grid's phase-major order."""
    if grid["encoding"] != ENCODING or grid["order"] != ORDER:
        raise ValueError("unsupported encoding or order")
    size = int(grid["phase_axis"]["count"]) * int(grid["delta_v_axis"]["count"])
    outcomes: list[str] = list(grid["outcomes"])
    codes = decode_codes(str(grid["codes"]), size)
    if max(codes, default=0) >= len(outcomes):
        raise ValueError("code outside the outcome table")
    return [outcomes[c] for c in codes]


def dumps_grid(grid: Mapping[str, Any]) -> bytes:
    """Compact JSON bytes (no NaN) with a trailing newline."""
    return (json.dumps(grid, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
