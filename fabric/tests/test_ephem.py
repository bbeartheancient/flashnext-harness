"""ephem gate: Standish Keplerian vs JPL Horizons 2026-08-25 00:00 UT.

Fixtures captured live from ssd.jpl.nasa.gov horizons.api
(geocentric apparent ICRF RA/DEC). Gate = angular separation < 1 deg;
measured max deviation was 0.083 deg.
"""

import math

from fabric.ephem import planets

_FIXTURES = {
    "Mars":    (6 + 37 / 60 + 19.96 / 3600, 23 + 35 / 60 + 23.1 / 3600),
    "Venus":   (12 + 58 / 60 + 29.90 / 3600, -(8 + 58 / 60 + 34.1 / 3600)),
    "Jupiter": (8 + 57 / 60 + 49.21 / 3600, 17 + 43 / 60 + 49.4 / 3600),
    "Saturn":  (0 + 54 / 60 + 22.78 / 3600, 2 + 58 / 60 + 7.3 / 3600),
}


def _sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2 + 180) % 360 - 180
    ddec = dec1 - dec2
    return math.hypot(dra * math.cos(math.radians((dec1 + dec2) / 2)), ddec)


def test_planets_match_horizons():
    out = planets("2026-08-25 00:00")
    by = {r["name"]: r for r in out["planets"]}
    for name, (ra_h, dec_d) in _FIXTURES.items():
        row = by[name]
        sep = _sep_deg(row["ra_deg"], row["dec_deg"], ra_h * 15, dec_d)
        assert sep < 1.0, f"{name} off by {sep:.3f} deg"


def test_earth_orbit_radius_sane():
    out = planets("2026-08-25 00:00")
    earth = next(r for r in out["planets"] if r["name"] == "Earth")
    assert 0.98 < earth["r_au"] < 1.02


def test_jd_known_value():
    from fabric.ephem import _jd_from_date

    assert abs(_jd_from_date("2000-01-01 12:00") - 2451545.0) < 1e-6


def test_orbits_sampled_shapes():
    from fabric.ephem import orbits_sampled

    out = orbits_sampled("2026-08-25", points=32)
    assert set(out["paths"]) >= {"Earth", "Jupiter"}
    for pts in out["paths"].values():
        assert len(pts) == 32
        r = math.sqrt(sum(v * v for v in pts[0]))
        assert r > 0.1
