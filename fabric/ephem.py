"""Solar-system ephemeris — Standish/JPL Keplerian approximation.

Valid ~1800–2050 AD. Elements + rates per Julian century from
JPL/Caltech "Approximate Positions of the Major Planets". Accuracy vs
published ephemerides is arcminutes-class; the unit test gates four
planets against JPL Horizons RA/DEC at 1 degree.

All angles in degrees until conversion. Output: heliocentric ecliptic
xyz (AU), ecliptic longitude, radius, plus geocentric RA/DEC (equatorial,
astrometric — no aberration/light-time; that is the 1-deg budget).
"""

from __future__ import annotations

import math

# name: (a, e, I, L, long.peri, long.node), rates per century
_ELEMENTS = {
    "Mercury": ((0.38709927, 0.20563593, 7.00497902, 252.25032350,
                 77.45779628, 48.33076593),
                (0.00000037, 0.00001906, -0.00594749, 149472.67411175,
                 0.16047689, -0.12534081)),
    "Venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950,
               131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729,
               0.00268329, -0.27769418)),
    "Earth": ((1.00000261, 0.01671123, -0.00001531, 100.46457166,
               102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981,
               0.32327364, 0.0)),
    "Mars": ((1.52371034, 0.09339410, 1.84969142, -4.55343205,
              -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499,
              0.44441088, -0.29257343)),
    "Jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051,
                 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775,
                 0.21252668, 0.20469106)),
    "Saturn": ((9.53667594, 0.05386179, 2.48599187, 49.95424423,
                92.59887831, 113.66242448),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201,
                -0.41897216, -0.28867794)),
    "Uranus": ((19.18916464, 0.04725744, 0.77263783, 313.23810451,
                170.95427630, 74.01692503),
               (-0.00196176, -0.00004397, -0.00242939, 428.48202785,
                0.40805281, 0.04240589)),
    "Neptune": ((30.06992276, 0.00859048, 1.77004347, -55.12002969,
                 44.96476227, 131.78422574),
                (0.00026291, 0.00005105, 0.00035372, 218.45945325,
                 0.32241464, -0.00508664)),
}

_OBLIQUITY_DEG = 23.43928


def _jd_from_date(date_iso: str) -> float:
    """date 'YYYY-MM-DD[ HH:MM]' -> Julian Date (UT assumed)."""
    date_part, _, time_part = (date_iso or "").strip().partition(" ")
    y, m, d = (int(x) for x in date_part.split("-")[:3])
    hh = mm = 0.0
    if time_part:
        bits = time_part.split(":")
        hh = float(bits[0] or 0)
        mm = float(bits[1]) if len(bits) > 1 else 0.0
    if m <= 2:
        y -= 1
        m += 12
    a = math.floor(y / 100)
    b = 2 - a + math.floor(a / 4)
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d + b - 1524.5 + (hh + mm / 60.0) / 24.0)


def _norm360(x: float) -> float:
    return x % 360.0


def _solve_kepler(m_deg: float, e: float) -> float:
    m_rad = math.radians(m_deg)
    ecc = math.radians(e * 180.0 / math.pi) * 1.0  # small-angle seed in rad
    ecc = m_rad
    for _ in range(8):
        delta = (ecc - e * math.sin(ecc) - m_rad) / (1 - e * math.cos(ecc))
        ecc -= delta
        if abs(delta) < 1e-10:
            break
    return ecc


def _helio(name: str, jd: float) -> tuple[float, float, float, float]:
    elems, rates = _ELEMENTS[name]
    t = (jd - 2451545.0) / 36525.0
    a = elems[0] + rates[0] * t
    e = elems[1] + rates[1] * t
    inc = math.radians(elems[2] + rates[2] * t)
    lon = _norm360(elems[3] + rates[3] * t)
    peri = _norm360(elems[4] + rates[4] * t)
    node = _norm360(elems[5] + rates[5] * t)

    arg_peri = math.radians(_norm360(peri - node))
    mean_anom = _norm360(lon - peri)
    if mean_anom > 180:
        mean_anom -= 360
    eccentric = _solve_kepler(mean_anom, e)

    xp = a * (math.cos(eccentric) - e)
    yp = a * math.sqrt(1 - e * e) * math.sin(eccentric)

    cos_o = math.cos(arg_peri)
    sin_o = math.sin(arg_peri)
    cos_n = math.cos(math.radians(node))
    sin_n = math.sin(math.radians(node))
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    x = (cos_o * cos_n - sin_o * sin_n * cos_i) * xp \
        + (-sin_o * cos_n - cos_o * sin_n * cos_i) * yp
    y = (cos_o * sin_n + sin_o * cos_n * cos_i) * xp \
        + (-sin_o * sin_n + cos_o * cos_n * cos_i) * yp
    z = sin_o * sin_i * xp + cos_o * sin_i * yp
    r = math.sqrt(xp * xp + yp * yp)
    return x, y, z, r


def _equatorial(vec) -> tuple[float, float]:
    eps = math.radians(_OBLIQUITY_DEG)
    x, y, z = vec
    xe = x
    ye = y * math.cos(eps) - z * math.sin(eps)
    ze = y * math.sin(eps) + z * math.cos(eps)
    ra = math.degrees(math.atan2(ye, xe)) % 360.0
    dec = math.degrees(math.atan2(ze, math.hypot(xe, ye)))
    return ra, dec


def planets(date_iso: str = "") -> dict:
    """Positions for all planets at date ('YYYY-MM-DD [HH:MM]', now default)."""
    import datetime as _dt

    if not date_iso or not date_iso.strip():
        now = _dt.datetime.now(_dt.timezone.utc)
        date_iso = now.strftime("%Y-%m-%d %H:%M")
    jd = _jd_from_date(date_iso)
    rows = []
    coords = {}
    for name in _ELEMENTS:
        x, y, z, r = _helio(name, jd)
        coords[name] = (x, y, z)
        rows.append({"name": name, "x": round(x, 5), "y": round(y, 5),
                     "z": round(z, 5), "r_au": round(r, 5),
                     "lon_deg": round(math.degrees(math.atan2(y, x)) % 360.0,
                                      3)})
    out = {"date": date_iso, "jd": round(jd, 5), "planets": rows}
    earth = coords.get("Earth")
    if earth:
        for row in rows:
            if row["name"] == "Earth":
                continue
            px, py, pz = coords[row["name"]]
            geo = (px - earth[0], py - earth[1], pz - earth[2])
            ra, dec = _equatorial(geo)
            row["ra_deg"] = round(ra, 4)
            row["dec_deg"] = round(dec, 4)
            dist = math.sqrt(sum(v * v for v in geo))
            row["geo_dist_au"] = round(dist, 5)
    return out


def orbits_sampled(date_iso: str = "", points: int = 96) -> dict:
    """Orbit polylines (heliocentric xyz per planet) for the globe view."""
    import datetime as _dt

    if not date_iso or not date_iso.strip():
        now = _dt.datetime.now(_dt.timezone.utc)
        date_iso = now.strftime("%Y-%m-%d %H:%M")
    jd0 = _jd_from_date(date_iso)
    n = max(16, min(int(points), 512))
    paths = {}
    for name, (elems, rates) in _ELEMENTS.items():
        daily_motion = rates[3] / 36525.0
        if abs(daily_motion) < 1e-9:
            continue
        period_days = 360.0 / abs(daily_motion)
        pts = []
        for i in range(n):
            x, y, z, r = _helio(name, jd0 + period_days * i / n)
            pts.append([round(x, 4), round(y, 4), round(z, 4)])
        paths[name] = pts
    return {"date": date_iso, "paths": paths}
