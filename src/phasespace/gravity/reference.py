"""Published reference data used to validate the physics core.

Values below are copied verbatim from the NASA/JPL Three-Body Periodic Orbits API
(Poincare Catalog of Periodic Orbits), response to
https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=earth-moon&family=lyapunov&libr=1
retrieved 2026-10-18 (API signature version 1.0). The same values are stored, together
with other family members, in tests/gravity/fixtures/jpl_earth_moon_l1_lyapunov.json and
a test checks that both copies agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from phasespace.gravity.system import EARTH_MEAN_RADIUS_KM, EarthMoonSystem

JPL_PERIODIC_ORBITS_URL: Final = (
    "https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=earth-moon&family=lyapunov&libr=1"
)
JPL_RETRIEVED_ON: Final = "2026-10-18"

JPL_EARTH_MOON: Final = EarthMoonSystem(
    name="earth-moon-jpl-periodic-orbits-catalog",
    mu=1.215058560962404e-02,
    length_scale_km=389703.264829278,
    time_scale_s=382981.289129055,
    # The catalog gives only the secondary's radius; the Earth radius is the project's
    # NSSDCA volumetric mean radius (see phasespace.gravity.system).
    earth_radius_km=EARTH_MEAN_RADIUS_KM,
    moon_radius_km=1737.1,
    source=(
        f"mu, lunit, tunit, radius_secondary: NASA/JPL Three-Body Periodic Orbits API "
        f"({JPL_PERIODIC_ORBITS_URL}, retrieved {JPL_RETRIEVED_ON}); Earth radius 6371.0 km "
        "from the NASA NSSDCA Earth Fact Sheet"
    ),
)
"""Earth-Moon constants of the JPL periodic-orbit catalog (note lunit != 384 400 km)."""


@dataclass(frozen=True, slots=True)
class CatalogOrbit:
    """A planar symmetric periodic orbit as listed by the JPL catalog (normalized units)."""

    family: str
    libration_point: int
    row_index: int
    x0: float
    vy0: float
    jacobi: float
    period: float
    stability_index: float


JPL_L1_LYAPUNOV_REFERENCE: Final = CatalogOrbit(
    family="lyapunov",
    libration_point=1,
    row_index=2600,
    x0=8.1507244801212275e-01,
    vy0=2.1717137836681638e-01,
    jacobi=3.14618105591714,
    period=2.8624598979521072e00,
    stability_index=904.465406637648,
)
"""Earth-Moon L1 Lyapunov orbit used for the M1 reference export (row 2600 of 3108)."""
