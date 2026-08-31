"""Geo physics calculator — B70 RTU terrain functions + hoa64 Deygout.

The individual-function surface (python/afi/rtu.py, docs/hoa64-rtu-backend.md)
wrapped as typed, JSON-safe calculator calls for fabric tools and the LCARS
geo pane. Terrain is a real Mapzen Terrarium DEM tile (hoa64.site_survey
fetch + RGB decode) centred on the station; fBm from hoa64.terrain is the
offline fallback. Geometry from the B70 RTUs with a numpy ray-march fallback
so the surface degrades gracefully on machines without libafi_rtu.

Functions:
  terrain_meta()               -> {G, span, hmin, hmax, source, lat/lon, heightmap}
  load_origin(lat, lon, zoom)  -> swap the resident tile (rebuilds the BLAS)
  los(tx, rx, f_mhz)           -> {los_clear, diffraction_loss_db, ...}
  los_latlon(...)              -> same, endpoints as lat/lon + AGL
  survey(tx_lat, …)            -> multi-tile great-circle Terrarium profile
  horizon(center, h_agl, ...)  -> {azimuths_deg, distances_m, hit[]}
  geocode(q)                   -> Nominatim hits {lat, lon, name, display}
  show_place(q, view=…)        -> load DEM at a named place + heatmap/horizon
"""

import json
import math
import os
import sys
import threading
import urllib.parse
import urllib.request

import numpy as np


def _import_hoa64():
    """Import hoa64 from the sibling tree (AGENTS.md: import, don't vendor)."""
    try:
        import hoa64  # noqa: F401

        return True
    except ModuleNotFoundError:
        root = os.environ.get("HOA64_ROOT", os.path.expanduser("~"))
        if os.path.isdir(os.path.join(root, "hoa64")):
            sys.path.insert(0, root)
            import hoa64  # noqa: F401

            return True
        raise


# hoa64 site_survey self-check coordinates (West Midlands). Tile 12/2018/1344
# is already in ~/.cache/hoa64/terrain from the sibling lab.
_DEFAULT_LAT = 52.445472
_DEFAULT_LON = -2.597833
_DEFAULT_ZOOM = 12
_TERRAIN_SEED = 20260823
_FBM_G = 256
_FBM_SPAN = 4096.0
_FBM_HMAX = 60.0
_WGS84_A = 6378137.0

_lock = threading.Lock()
_terrain = None
_origin = None
_blas = None
_rtu_ok = None
_last_los = None


def _decode_terrarium(tile: np.ndarray) -> np.ndarray:
    t = np.asarray(tile, dtype=np.float32)
    return t[..., 0] * 256.0 + t[..., 1] + t[..., 2] / 256.0 - 32768.0


def heights_to_terrarium(h: np.ndarray) -> np.ndarray:
    """Pack a heightfield into Terrarium RGB (test fixture helper)."""
    x = np.clip(np.rint((np.asarray(h, dtype=np.float64) + 32768.0) * 256.0),
                0, 256 * 256 * 256 - 1).astype(np.int32)
    rgb = np.empty(h.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (x // 65536) & 255
    rgb[..., 1] = (x // 256) & 255
    rgb[..., 2] = x & 255
    return rgb


def _tile_span_m(lat: float, zoom: int) -> float:
    return (2.0 * math.pi * _WGS84_A
            * math.cos(math.radians(float(lat))) / float(1 << int(zoom)))


def _drop_blas():
    global _blas
    if _blas is not None:
        try:
            _blas.close()
        except Exception:
            pass
        _blas = None


def _fbm_fallback() -> np.ndarray:
    _import_hoa64()
    from hoa64.terrain import terrain as hoa_terrain

    t = hoa_terrain(size=_FBM_G, order=64, seed=_TERRAIN_SEED)
    return t["heightmap"].astype(np.float32) * _FBM_HMAX


def reset():
    """Drop resident terrain + BLAS (tests)."""
    global _terrain, _origin, _last_los
    with _lock:
        _drop_blas()
        _terrain = None
        _origin = None
        _last_los = None


def load_origin(lat: float = _DEFAULT_LAT, lon: float = _DEFAULT_LON,
                zoom: int = _DEFAULT_ZOOM, tile: np.ndarray | None = None) -> dict:
    """Resident DEM: one Terrarium tile centred on (lat, lon).

    `tile` injects a decoded RGB array (tests / offline). Network fetch goes
    through hoa64.site_survey.fetch_tile (cached under ~/.cache/hoa64/terrain).
    """
    global _terrain, _origin
    _import_hoa64()
    from hoa64.site_survey import _tile_pixel, fetch_tile

    lat, lon, zoom = float(lat), float(lon), int(zoom)
    tx, ty, _fx, _fy = _tile_pixel(lat, lon, zoom)
    if tile is None:
        tile = fetch_tile(zoom, tx, ty)
    h = np.ascontiguousarray(_decode_terrarium(tile), dtype=np.float32)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Terrarium tile must be square")
    span = _tile_span_m(lat, zoom)
    with _lock:
        _drop_blas()
        _terrain = h
        _origin = {
            "lat": lat,
            "lon": lon,
            "zoom": zoom,
            "tile_x": int(tx),
            "tile_y": int(ty),
            "span_m": float(span),
            "source": "terrarium",
            "G": int(h.shape[0] - 1),
        }
        return dict(_origin)


def _span() -> float:
    if _origin is not None:
        return float(_origin["span_m"])
    return _FBM_SPAN


def _heights() -> np.ndarray:
    global _terrain, _origin
    if _terrain is None:
        try:
            load_origin()
        except Exception:
            h = _fbm_fallback()
            with _lock:
                _terrain = h
                _origin = {
                    "lat": None, "lon": None, "zoom": None,
                    "tile_x": None, "tile_y": None,
                    "span_m": _FBM_SPAN, "source": "fbm",
                    "G": int(h.shape[0] - 1), "seed": _TERRAIN_SEED,
                }
    return _terrain


def _ground(h, x, z):
    G = h.shape[0] - 1
    s = G / _span()
    fi = np.clip(np.asarray(x) * s, 0, G - 1e-4)
    fj = np.clip(np.asarray(z) * s, 0, G - 1e-4)
    i = fi.astype(int)
    j = fj.astype(int)
    u = fi - i
    v = fj - j
    return (h[j, i] * (1 - u) * (1 - v) + h[j, i + 1] * u * (1 - v)
            + h[j + 1, i] * (1 - u) * v + h[j + 1, i + 1] * u * v)


def _rtu_available() -> bool:
    global _rtu_ok
    if _rtu_ok is None:
        try:
            repo_python = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "python")
            if os.path.isdir(repo_python) and repo_python not in sys.path:
                sys.path.insert(0, repo_python)
            from afi import rtu  # noqa: F401

            _rtu_ok = True
        except (ImportError, OSError):
            _rtu_ok = False
    return _rtu_ok


def _get_blas():
    global _blas
    if _blas is None:
        from afi import rtu

        rtu.init(int(os.environ.get("AFI_RTU_GPU", "1")))
        _blas = rtu.TerrainBLAS(_heights(), _span())
    return _blas


def _march_occlusion(rays6, tmax, step_m=1.0):
    """numpy fallback (approximate, 1 m step) — rtu_baselines arm."""
    h = _heights()
    org = rays6[:, 0:3]
    d = rays6[:, 3:6]
    n = len(rays6)
    out = np.zeros(n, dtype=bool)
    for c0 in range(0, n, 2048):
        c1 = min(c0 + 2048, n)
        steps = np.maximum((tmax[c0:c1] / step_m).astype(int), 2)
        smax = int(steps.max())
        t = np.arange(1, smax + 1)[None, :].astype(np.float32) * step_m
        t = np.minimum(t, tmax[c0:c1][:, None] - 1e-3)
        px = org[c0:c1, 0][:, None] + d[c0:c1, 0][:, None] * t
        py = org[c0:c1, 1][:, None] + d[c0:c1, 1][:, None] * t
        pz = org[c0:c1, 2][:, None] + d[c0:c1, 2][:, None] * t
        valid = t < tmax[c0:c1][:, None] - 1e-2
        g = _ground(h, px, pz)
        out[c0:c1] = ((py < g) & valid).any(axis=1)
    return out


def _march_horizon(rays6, tmax, step_m=2.0):
    """numpy fallback horizon: first sample where ground rises above ray."""
    h = _heights()
    org = rays6[:, 0:3]
    d = rays6[:, 3:6]
    n = len(rays6)
    out = np.full(n, -1.0, dtype=np.float32)
    steps = np.maximum((tmax / step_m).astype(int), 2)
    smax = int(steps.max())
    t = np.arange(1, smax + 1)[None, :].astype(np.float32) * step_m
    t = np.minimum(t, tmax[:, None] - 1e-3)
    px = org[:, 0][:, None] + d[:, 0][:, None] * t
    py = org[:, 1][:, None] + d[:, 1][:, None] * t
    pz = org[:, 2][:, None] + d[:, 2][:, None] * t
    g = _ground(h, px, pz)
    hit = (py < g) & (t < tmax[:, None] - 1e-2)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    rows = np.nonzero(first >= 0)[0]
    out[rows] = t[rows, first[rows]]
    return out


def terrain_meta() -> dict:
    """Downsampled heightmap + extents for the visualizer."""
    h = _heights()
    origin = dict(_origin or {})
    G = int(h.shape[0] - 1)
    step = max(1, (G + 1) // 128)
    ds = h[::step, ::step]
    return {
        "G": G,
        "span_m": round(_span(), 1),
        "hmin_m": round(float(h.min()), 2),
        "hmax_m": round(float(h.max()), 2),
        "source": origin.get("source"),
        "lat": origin.get("lat"),
        "lon": origin.get("lon"),
        "zoom": origin.get("zoom"),
        "tile_x": origin.get("tile_x"),
        "tile_y": origin.get("tile_y"),
        "seed": origin.get("seed"),
        "engine": "rtu" if _rtu_available() else "numpy-march",
        "heightmap": [[round(float(v), 2) for v in row] for row in ds],
    }


def los(tx, rx, f_mhz: float = 5800.0) -> dict:
    """TX->RX link check: RTU 3D occlusion + hoa64 Deygout profile terms."""
    tx = [float(v) for v in tx]
    rx = [float(v) for v in rx]
    h = _heights()
    org = np.asarray(tx, dtype=np.float32)
    end = np.asarray(rx, dtype=np.float32)
    d = end - org
    L = float(np.linalg.norm(d))
    if L < 1.0:
        return {"error": "degenerate segment"}
    u = d / L

    rays = np.concatenate([org, u])[None, :].astype(np.float32)
    tmax = np.array([L], dtype=np.float32)
    if _rtu_available():
        occluded = bool(_get_blas().occlusion(rays, tmax)[0])
        engine = "rtu"
    else:
        occluded = bool(_march_occlusion(rays, tmax)[0])
        engine = "numpy-march"

    # Deygout terms on the sampled profile (hoa64, pure geometry + ITU)
    diffraction_db = None
    min_clearance_m = None
    min_fresnel_frac = None
    try:
        from hoa64 import site_survey as ss

        ts = np.linspace(0.0, L, 256)
        px = org[0] + u[0] * ts
        py = org[1] + u[1] * ts
        pz = org[2] + u[2] * ts
        elev = _ground(h, px, pz)
        tx_h = float(org[1] - _ground(h, org[0], org[2]))
        rx_h = float(end[1] - _ground(h, end[0], end[2]))
        res = ss._analyze(ts, elev, tx_h=tx_h, rx_h=rx_h,
                          f_hz=float(f_mhz) * 1e6)
        diffraction_db = round(float(res["diffraction_loss_db"]), 2)
        min_clearance_m = round(float(res["clearance_m"]), 2)
        min_fresnel_frac = round(float(res["min_fresnel_clearance"]), 3)
        los_hoa64 = bool(res["los"])
    except Exception as e:  # noqa: BLE001
        los_hoa64 = None
        err = str(e)
    else:
        err = None

    if occluded:
        verdict = "BLOCKED"
    elif diffraction_db is not None and diffraction_db > 6.0:
        verdict = "CLEAR (marginal Fresnel)"
    else:
        verdict = "CLEAR"

    out = {
        "tx_m": [round(tx[0], 1), round(tx[1], 1), round(tx[2], 1)],
        "rx_m": [round(rx[0], 1), round(rx[1], 1), round(rx[2], 1)],
        "path_m": round(L, 1),
        "f_mhz": float(f_mhz),
        "los_clear": not occluded,
        "verdict": verdict,
        "engine": engine,
        "diffraction_loss_db": diffraction_db,
        "min_clearance_m": min_clearance_m,
        "min_fresnel_frac": min_fresnel_frac,
    }
    if los_hoa64 is not None:
        out["los_hoa64_deygout"] = los_hoa64
    if err:
        out["deygout_error"] = err
    if _origin:
        out["source"] = _origin.get("source")
        out["lat"] = _origin.get("lat")
        out["lon"] = _origin.get("lon")

    global _last_los
    with _lock:
        _last_los = out
    return out


def los_latlon(tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
               tx_h: float = 15.0, rx_h: float = 15.0,
               f_mhz: float = 5800.0) -> dict:
    """LOS with geographic endpoints. Both must fall inside the resident tile.

    tx_h / rx_h are antenna heights AGL (metres). Local x increases east,
    z increases south (Web Mercator tile row).
    """
    _heights()
    if not _origin or _origin.get("source") != "terrarium":
        return {"error": "resident terrain is not a Terrarium tile"}
    from hoa64.site_survey import _tile_pixel

    z = int(_origin["zoom"])
    tx_x, tx_y, tx_fx, tx_fy = _tile_pixel(tx_lat, tx_lon, z)
    rx_x, rx_y, rx_fx, rx_fy = _tile_pixel(rx_lat, rx_lon, z)
    if (tx_x, tx_y) != (_origin["tile_x"], _origin["tile_y"]) or (
            rx_x, rx_y) != (_origin["tile_x"], _origin["tile_y"]):
        return survey(tx_lat, tx_lon, rx_lat, rx_lon, tx_h, rx_h, f_mhz)
    span = _span()
    h = _heights()
    tx_wx, tx_wz = tx_fx / 256.0 * span, tx_fy / 256.0 * span
    rx_wx, rx_wz = rx_fx / 256.0 * span, rx_fy / 256.0 * span
    tx_wy = float(_ground(h, tx_wx, tx_wz)) + float(tx_h)
    rx_wy = float(_ground(h, rx_wx, rx_wz)) + float(rx_h)
    out = los([tx_wx, tx_wy, tx_wz], [rx_wx, rx_wy, rx_wz], f_mhz)
    out["tx_ll"] = [round(float(tx_lat), 6), round(float(tx_lon), 6),
                    round(float(tx_h), 2)]
    out["rx_ll"] = [round(float(rx_lat), 6), round(float(rx_lon), 6),
                    round(float(rx_h), 2)]
    return out


def last_los() -> dict | None:
    with _lock:
        return dict(_last_los) if _last_los else None


def _downsample_profile(arr, n=64):
    a = list(arr)
    if len(a) <= n:
        return [round(float(v), 2) for v in a]
    step = max(1, len(a) // n)
    return [round(float(v), 2) for v in a[::step]]


def _in_resident_tile(lat, lon) -> bool:
    if not _origin or _origin.get("source") != "terrarium":
        return False
    from hoa64.site_survey import _tile_pixel
    x, y, _, _ = _tile_pixel(float(lat), float(lon), int(_origin["zoom"]))
    return (x, y) == (_origin["tile_x"], _origin["tile_y"])


def survey(tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
           tx_h: float = 15.0, rx_h: float = 15.0, f_mhz: float = 5800.0,
           n: int = 200, zoom: int | None = None) -> dict:
    """Great-circle Terrarium survey (hoa64.site_survey). Multi-tile OK.

    When both ends sit on the resident tile, the RTU LOS is attached as
    `rtu`. Profile arrays are downsampled for the agent/pane.
    """
    import time as _time

    _import_hoa64()
    from hoa64.site_survey import survey as hoa_survey

    _heights()
    z = int(zoom or (_origin or {}).get("zoom") or _DEFAULT_ZOOM)
    t0 = _time.perf_counter()
    raw = hoa_survey(
        {"lat": float(tx_lat), "lon": float(tx_lon), "h_m": float(tx_h)},
        {"lat": float(rx_lat), "lon": float(rx_lon), "h_m": float(rx_h)},
        float(f_mhz), n=int(n), zoom=z,
    )
    ms = (_time.perf_counter() - t0) * 1e3
    out = {
        "kind": "survey",
        "engine": "terrarium-profile",
        "survey_ms": round(ms, 1),
        "tx": raw.get("tx"),
        "rx": raw.get("rx"),
        "path_m": round(float(raw.get("path_m", 0.0)), 1),
        "f_mhz": float(f_mhz),
        "zoom": int(raw.get("zoom", z)),
        "los": bool(raw.get("los")),
        "verdict": raw.get("verdict"),
        "diffraction_loss_db": round(float(raw.get("diffraction_loss_db", 0)), 2),
        "min_clearance_m": round(float(raw.get("clearance_m", 0)), 2),
        "min_fresnel_frac": round(float(raw.get("min_fresnel_clearance", 0)), 3),
        "received_dbw": None if raw.get("received_dbw") is None
        else round(float(raw["received_dbw"]), 2),
        "site_only": bool(raw.get("site_only")),
        "n_profile": len(raw.get("dist_m") or []),
        "dist_m": _downsample_profile(raw.get("dist_m") or []),
        "elev_m": _downsample_profile(raw.get("elev_m") or []),
        "multi_tile": not (
            _in_resident_tile(tx_lat, tx_lon) and _in_resident_tile(rx_lat, rx_lon)
        ),
        "figure": {
            "type": "xy",
            "x": _downsample_profile(raw.get("dist_m") or []),
            "y": _downsample_profile(raw.get("elev_m") or []),
            "xlabel": "path m",
            "ylabel": "elev m",
            "caption": (
                f"survey {round(float(raw.get('path_m', 0.0)), 1)} m · "
                f"{raw.get('verdict')}"
            ),
        },
    }
    if _in_resident_tile(tx_lat, tx_lon) and _in_resident_tile(rx_lat, rx_lon):
        from hoa64.site_survey import _tile_pixel
        span = _span()
        h = _heights()
        z0 = int(_origin["zoom"])
        _, _, tx_fx, tx_fy = _tile_pixel(tx_lat, tx_lon, z0)
        _, _, rx_fx, rx_fy = _tile_pixel(rx_lat, rx_lon, z0)
        tx_wx, tx_wz = tx_fx / 256.0 * span, tx_fy / 256.0 * span
        rx_wx, rx_wz = rx_fx / 256.0 * span, rx_fy / 256.0 * span
        rtu = los(
            [tx_wx, float(_ground(h, tx_wx, tx_wz)) + float(tx_h), tx_wz],
            [rx_wx, float(_ground(h, rx_wx, rx_wz)) + float(rx_h), rx_wz],
            f_mhz,
        )
        out["rtu"] = {
            "los_clear": rtu.get("los_clear"),
            "verdict": rtu.get("verdict"),
            "engine": rtu.get("engine"),
            "path_m": rtu.get("path_m"),
        }
        out["tx_m"] = rtu.get("tx_m")
        out["rx_m"] = rtu.get("rx_m")
        out["los_clear"] = rtu.get("los_clear")
    else:
        out["los_clear"] = bool(raw.get("los"))
    global _last_los
    with _lock:
        _last_los = out
    return out


def horizon(center, h_agl: float, n_az: int = 360,
            elev_deg: float = -2.0) -> dict:
    """Azimuth horizon sweep from a sensor: first-hit distance per azimuth."""
    cx, cz = float(center[0]), float(center[1])
    h = _heights()
    h0 = float(_ground(h, cx, cz)) + float(h_agl)
    az = np.linspace(0.0, 360.0, int(n_az), endpoint=False)
    e = math.radians(float(elev_deg))
    dirs = np.stack([np.cos(np.deg2rad(az)), np.full(len(az), math.sin(e)),
                     np.sin(np.deg2rad(az))], axis=1).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    org = np.tile(np.array([cx, h0, cz], dtype=np.float32), (len(az), 1))
    rays = np.concatenate([org, dirs], axis=1).astype(np.float32)
    tmax = np.full(len(az), 2.0 * _span(), dtype=np.float32)
    if _rtu_available():
        t, _prim = _get_blas().horizon(rays, tmax)
        engine = "rtu"
    else:
        t = _march_horizon(rays, tmax)
        engine = "numpy-march"
    az_list = [round(float(a), 2) for a in az]
    dist = [round(float(v), 1) for v in t]
    mx = max(dist) or 1.0
    return {
        "center_m": [round(cx, 1), round(cz, 1)],
        "sensor_h_m": round(h0, 1),
        "elev_deg": float(elev_deg),
        "azimuths_deg": az_list,
        "distances_m": dist,
        "engine": engine,
        "figure": {
            "type": "polar",
            "theta_deg": az_list,
            "power_norm": [round(d / mx, 4) for d in dist],
            "caption": f"horizon {engine} · max {mx:.0f} m",
        },
    }


_geocode_cache: dict[str, list] = {}


_nominatim_last = [0.0]


def _nominatim_search(q: str, limit: int = 3) -> list:
    """Network fetch. Tests monkeypatch this (do not hit OSM in pytest).

    Paced to the OSM usage policy (~1 req/s); bursts get soft-blocked
    and look like empty results."""
    import time as _t

    wait = 1.1 - (_t.time() - _nominatim_last[0])
    if wait > 0:
        _t.sleep(wait)
    _nominatim_last[0] = _t.time()
    qs = urllib.parse.urlencode({"q": q, "format": "json", "limit": int(limit)})
    req = urllib.request.Request(
        "https://nominatim.openstreetmap.org/search?" + qs,
        headers={"User-Agent": "woodfire-fabric/0.1 (local ship computer)"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode())


def geocode(q: str, limit: int = 3) -> list[dict]:
    """Resolve a place name to lat/lon. Cached. Empty list on failure."""
    q = (q or "").strip()
    if not q:
        return []
    key = q.lower()
    if key in _geocode_cache:
        return list(_geocode_cache[key])
    try:
        raw = _nominatim_search(q, limit)
    except Exception:
        return []
    hits = []
    for row in raw or []:
        try:
            hits.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "name": row.get("name") or q,
                "display": row.get("display_name") or row.get("name") or q,
                "kind": row.get("type") or row.get("class"),
            })
        except (KeyError, TypeError, ValueError):
            continue
    _geocode_cache[key] = hits
    return list(hits)


def show_place(q: str = "", lat: float | None = None, lon: float | None = None,
               zoom: int = 12, view: str = "terrain",
               h_agl: float = 25.0, n_az: int = 72) -> dict:
    """Load the resident DEM on a named place (or explicit lat/lon).

    Default view is the heightmap heatmap (terrain slice). `horizon` runs
    the RTU/numpy coverage fan from the tile centre after the origin swap.
    Multiple geocode hits return clarify choices instead of guessing.
    """
    hits = []
    if lat is not None and lon is not None:
        hits = [{"lat": float(lat), "lon": float(lon),
                 "name": q or f"{lat:.4f},{lon:.4f}",
                 "display": q or f"{lat:.4f},{lon:.4f}"}]
    elif q:
        hits = geocode(q)
    if not hits:
        return {
            "error": "no geocode",
            "clarify": True,
            "query": q,
            "choices": [
                {"kind": "fn", "id": "place", "label": "enter lat/lon and Apply"},
                {"kind": "fn", "id": "horizon", "label": "horizon on current tile"},
                {"kind": "fn", "id": "survey", "label": "terrain survey (two antennas)"},
            ],
            "hint": "geocode missed — type lat/lon on the left rail, or pick a library",
        }
    if lat is None and lon is None and len(hits) > 1:
        return {
            "clarify": True,
            "query": q,
            "choices": [
                {"kind": "place", "id": h["display"],
                 "label": h["display"], "lat": h["lat"], "lon": h["lon"]}
                for h in hits[:5]
            ],
            "hint": "several places match — pick one",
            "hits": hits[:5],
        }
    hit = hits[0]
    try:
        origin = load_origin(hit["lat"], hit["lon"], int(zoom or 12))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "lat": hit["lat"], "lon": hit["lon"],
                "name": hit["name"]}
    view = (view or "terrain").lower()
    if view == "horizon":
        meta = terrain_meta()
        half = float(meta["span_m"]) / 2.0
        out = horizon([half, half], h_agl, n_az)
        out["name"] = hit["name"]
        out["display"] = hit.get("display")
        out["lat"] = hit["lat"]
        out["lon"] = hit["lon"]
        out["origin"] = origin
        if out.get("figure"):
            out["figure"]["caption"] = f"{hit['name']} horizon"
        return out
    meta = terrain_meta()
    meta["name"] = hit["name"]
    meta["display"] = hit.get("display")
    meta["origin"] = origin
    meta["figure"] = {
        "type": "heatmap",
        "grid": meta.get("heightmap") or [],
        "caption": f"{hit['name']} terrain",
    }
    return meta


def geocode_fast(q: str, limit: int = 2) -> list[dict]:
    """Open-Meteo geocoder — no key, lenient rate limits. Same shape
    as geocode(); used by batch jobs (news) where Nominatim policy
    would throttle."""
    q = (q or "").strip()
    if not q:
        return []
    qs = urllib.parse.urlencode({"name": q, "count": int(limit),
                                 "language": "en", "format": "json"})
    req = urllib.request.Request(
        "https://geocoding-api.open-meteo.com/v1/search?" + qs,
        headers={"User-Agent": "woodfire-fabric/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    hits = []
    for row in data.get("results") or []:
        try:
            # normalize GEO feature codes -> friendly kinds
            fc = str(row.get("feature_code") or "").upper()
            kind = ("city" if fc.startswith("P")
                    else "administrative" if fc.startswith("A")
                    else "place")
            hits.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "name": row.get("name") or q,
                "display": ", ".join(x for x in (
                    row.get("name"), row.get("admin1"),
                    row.get("country")) if x),
                "kind": kind,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return hits
