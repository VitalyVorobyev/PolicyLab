"""Loader for the stored JPL Three-Body Periodic Orbits API fixture (no network access)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from phasespace.gravity.system import EARTH_MEAN_RADIUS_KM, EarthMoonSystem

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "jpl_earth_moon_l1_lyapunov.json"


@dataclass(frozen=True)
class JplOrbit:
    row_index: int
    x0: float
    y0: float
    vx0: float
    vy0: float
    jacobi: float
    period: float
    stability: float


@dataclass(frozen=True)
class JplCatalog:
    source_url: str
    fetched_on: str
    mu: float
    lunit_km: float
    tunit_s: float
    radius_secondary_km: float
    l1: float
    l2: float
    l3: float
    orbits: tuple[JplOrbit, ...]

    def system(self) -> EarthMoonSystem:
        """The catalog's constants as an explicit system (Earth radius: project default)."""
        return EarthMoonSystem(
            name="jpl-fixture",
            mu=self.mu,
            length_scale_km=self.lunit_km,
            time_scale_s=self.tunit_s,
            earth_radius_km=EARTH_MEAN_RADIUS_KM,
            moon_radius_km=self.radius_secondary_km,
            source=self.source_url,
        )


def load_catalog() -> JplCatalog:
    raw = cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
    system = cast(dict[str, Any], raw["system"])
    fields = cast(list[str], raw["fields"])
    orbits: list[JplOrbit] = []
    for entry in cast(list[dict[str, Any]], raw["orbits"]):
        values = dict(zip(fields, (float(v) for v in entry["values"]), strict=True))
        orbits.append(
            JplOrbit(
                row_index=int(entry["row_index"]),
                x0=values["x"],
                y0=values["y"],
                vx0=values["vx"],
                vy0=values["vy"],
                jacobi=values["jacobi"],
                period=values["period"],
                stability=values["stability"],
            )
        )
    return JplCatalog(
        source_url=str(raw["source_url"]),
        fetched_on=str(raw["fetched_on"]),
        mu=float(system["mass_ratio"]),
        lunit_km=float(system["lunit"]),
        tunit_s=float(system["tunit"]),
        radius_secondary_km=float(system["radius_secondary"]),
        l1=float(system["L1"][0]),
        l2=float(system["L2"][0]),
        l3=float(system["L3"][0]),
        orbits=tuple(orbits),
    )
