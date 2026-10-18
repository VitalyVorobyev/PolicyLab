"""Earth-Moon system definitions for the normalized planar CR3BP.

A system is always passed explicitly to the functions in this package, so a reference
that uses different constants (for example the JPL periodic-orbit catalog) can be
reproduced exactly by constructing an `EarthMoonSystem` with that reference's values.

Normalized units: length unit = distance between the primaries, time unit = inverse of
the mean angular rate of the primaries, so that the angular rate and the primaries'
separation are both 1. The velocity unit is length unit / time unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, get_args

type LengthScaleConvention = Literal["nssdca_lunar_semimajor_axis", "jpl_periodic_orbits_catalog"]
"""Which length unit km values refer to: the 384 400 km NSSDCA lunar semimajor axis, or
the JPL periodic-orbit catalog's lunit (389 703.264829278 km)."""

LENGTH_SCALE_CONVENTIONS: Final[tuple[str, ...]] = get_args(LengthScaleConvention.__value__)

# Gravitational parameters from the JPL planetary ephemeris DE440 as listed on
# https://ssd.jpl.nasa.gov/astro_par.html (retrieved 2026-10-18); Park et al. (2021),
# "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161:105.
GM_EARTH_KM3_S2 = 398600.435507
GM_MOON_KM3_S2 = 4902.800118

# Semimajor axis of the lunar orbit, 0.3844e6 km, from the NASA NSSDCA Moon Fact Sheet
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html (retrieved 2026-10-18).
EARTH_MOON_DISTANCE_KM = 384400.0

# Volumetric mean radii from the NASA NSSDCA Earth and Moon Fact Sheets
# (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html and moonfact.html,
# retrieved 2026-10-18).
EARTH_MEAN_RADIUS_KM = 6371.0
MOON_MEAN_RADIUS_KM = 1737.4


@dataclass(frozen=True, slots=True)
class EarthMoonSystem:
    """Constants of a planar Earth-Moon CR3BP model.

    Attributes:
        name: Short identifier recorded in exported artifacts.
        mu: Mass parameter m_moon / (m_earth + m_moon).
        length_scale_km: Length unit (Earth-Moon distance) in km.
        time_scale_s: Time unit (inverse angular rate) in seconds.
        earth_radius_km: Earth radius used for default collision detection.
        moon_radius_km: Moon radius used for default collision detection.
        source: Human-readable provenance of the constants.
        length_scale_convention: Machine-readable name of the length-unit convention.
    """

    name: str
    mu: float
    length_scale_km: float
    time_scale_s: float
    earth_radius_km: float
    moon_radius_km: float
    source: str
    length_scale_convention: LengthScaleConvention

    def __post_init__(self) -> None:
        if self.length_scale_convention not in LENGTH_SCALE_CONVENTIONS:
            raise ValueError(f"unknown length_scale_convention {self.length_scale_convention!r}")
        if not (0.0 < self.mu <= 0.5):
            raise ValueError(f"mu must be in (0, 0.5], got {self.mu}")
        for label, value in (
            ("length_scale_km", self.length_scale_km),
            ("time_scale_s", self.time_scale_s),
            ("earth_radius_km", self.earth_radius_km),
            ("moon_radius_km", self.moon_radius_km),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be finite and positive, got {value}")

    @property
    def velocity_scale_km_s(self) -> float:
        """Velocity unit in km/s (length unit / time unit)."""
        return self.length_scale_km / self.time_scale_s

    @property
    def earth_position(self) -> tuple[float, float]:
        """Earth position (x, y) in the normalized barycentric rotating frame."""
        return (-self.mu, 0.0)

    @property
    def moon_position(self) -> tuple[float, float]:
        """Moon position (x, y) in the normalized barycentric rotating frame."""
        return (1.0 - self.mu, 0.0)


def system_from_gm(
    *,
    name: str,
    gm_earth_km3_s2: float,
    gm_moon_km3_s2: float,
    length_scale_km: float,
    earth_radius_km: float,
    moon_radius_km: float,
    source: str,
    length_scale_convention: LengthScaleConvention,
) -> EarthMoonSystem:
    """Build a system whose mu and time unit follow from GM values and the length unit.

    mu = GM_moon / (GM_earth + GM_moon) and, from Kepler's third law for a circular
    relative orbit, time_scale = sqrt(L^3 / (GM_earth + GM_moon)).
    """
    gm_total = gm_earth_km3_s2 + gm_moon_km3_s2
    return EarthMoonSystem(
        name=name,
        mu=gm_moon_km3_s2 / gm_total,
        length_scale_km=length_scale_km,
        time_scale_s=math.sqrt(length_scale_km**3 / gm_total),
        earth_radius_km=earth_radius_km,
        moon_radius_km=moon_radius_km,
        source=source,
        length_scale_convention=length_scale_convention,
    )


EARTH_MOON = system_from_gm(
    name="earth-moon-de440-384400km",
    gm_earth_km3_s2=GM_EARTH_KM3_S2,
    gm_moon_km3_s2=GM_MOON_KM3_S2,
    length_scale_km=EARTH_MOON_DISTANCE_KM,
    earth_radius_km=EARTH_MEAN_RADIUS_KM,
    moon_radius_km=MOON_MEAN_RADIUS_KM,
    source=(
        "GM_earth, GM_moon: JPL DE440 via https://ssd.jpl.nasa.gov/astro_par.html; "
        "length unit 384400 km and mean radii: NASA NSSDCA Earth/Moon Fact Sheets; "
        "time unit derived as sqrt(L^3 / (GM_earth + GM_moon)); retrieved 2026-10-18"
    ),
    length_scale_convention="nssdca_lunar_semimajor_axis",
)
"""Project default Earth-Moon system (DE440 GMs, 384 400 km length unit)."""
