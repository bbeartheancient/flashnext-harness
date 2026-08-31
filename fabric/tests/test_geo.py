"""Tests for the fabric geo calculator (fabric/geo.py + endpoints).

Uses an injected synthetic Terrarium ridge so the suite does not need the
network. A separate test loads the cached station tile when present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fabric import geo
except Exception as e:  # noqa: BLE001
    pytest.skip(f"geo surface unavailable: {e}", allow_module_level=True)


def _ridge_tile():
    h = np.full((256, 256), 10.0, dtype=np.float32)
    i = np.arange(256, dtype=np.float32)
    h += (80.0 * np.exp(-((i - 128.0) / 10.0) ** 2))[None, :]
    return geo.heights_to_terrarium(h), h


@pytest.fixture()
def ridge():
    geo.reset()
    tile, h = _ridge_tile()
    geo.load_origin(52.445472, -2.597833, 12, tile=tile)
    yield h
    geo.reset()


def test_terrain_meta_shape(ridge):
    meta = geo.terrain_meta()
    assert meta["source"] == "terrarium"
    assert meta["G"] == 255
    assert meta["span_m"] > 1000
    assert len(meta["heightmap"]) == 128
    assert len(meta["heightmap"][0]) == 128
    assert meta["hmax_m"] > meta["hmin_m"]
    assert meta["engine"] in ("rtu", "numpy-march")
    assert meta["lat"] == pytest.approx(52.445472)
    assert meta["tile_x"] == 2018


def test_los_clear_high_altitude(ridge):
    meta = geo.terrain_meta()
    span = meta["span_m"]
    y = meta["hmax_m"] + 80.0
    res = geo.los([0.05 * span, y, 0.05 * span],
                  [0.95 * span, y, 0.95 * span])
    assert res["los_clear"] is True
    assert res["verdict"].startswith("CLEAR")


def test_los_blocked_across_ridge(ridge):
    span = geo.terrain_meta()["span_m"]
    # 3 m AGL either side of the N-S ridge at x=span/2
    res = geo.los([0.25 * span, 13.0, 0.5 * span],
                  [0.75 * span, 13.0, 0.5 * span])
    assert res["los_clear"] is False
    assert res["verdict"] == "BLOCKED"


def test_los_degenerate(ridge):
    assert "error" in geo.los([100, 50, 100], [100.5, 50, 100])


def test_los_stores_last(ridge):
    span = geo.terrain_meta()["span_m"]
    res = geo.los([0.1 * span, 200, 0.1 * span],
                  [0.9 * span, 200, 0.9 * span])
    assert geo.last_los() == res


def test_horizon_shape(ridge):
    span = geo.terrain_meta()["span_m"]
    res = geo.horizon([span / 2, span / 2], h_agl=40.0, n_az=72)
    assert len(res["azimuths_deg"]) == 72
    assert len(res["distances_m"]) == 72
    hits = [d for d in res["distances_m"] if d > 0]
    assert 0 < len(hits) <= 72
    assert res["engine"] in ("rtu", "numpy-march")


def test_los_latlon_inside_tile(ridge):
    # tile 12/2018/1344 contains 52.445472, -2.597833
    res = geo.los_latlon(52.447, -2.60, 52.444, -2.59, tx_h=200, rx_h=200)
    if "error" in res:
        pytest.skip(res["error"])
    assert "tx_ll" in res and "verdict" in res


def test_show_place_heatmap(ridge, monkeypatch):
    monkeypatch.setattr(geo, "geocode", lambda q, limit=3: [{
        "lat": 52.445472, "lon": -2.597833, "name": "Neosho",
        "display": "Neosho, MO, USA",
    }])
    monkeypatch.setattr(geo, "load_origin", lambda lat, lon, zoom=12, tile=None: {
        "lat": lat, "lon": lon, "zoom": zoom, "source": "terrarium",
    })
    out = geo.show_place("neosho, mo")
    assert out.get("clarify") is not True
    assert out["figure"]["type"] == "heatmap"
    assert "Neosho" in out["figure"]["caption"]
    assert out["hmax_m"] > out["hmin_m"]


def test_show_place_clarify_multi(monkeypatch):
    monkeypatch.setattr(geo, "geocode", lambda q, limit=3: [
        {"lat": 1.0, "lon": 2.0, "name": "A", "display": "A, USA"},
        {"lat": 3.0, "lon": 4.0, "name": "B", "display": "B, USA"},
    ])
    out = geo.show_place("springfield")
    assert out["clarify"] is True
    assert len(out["choices"]) == 2


def test_tools_registered():
    from fabric.tools import TOOLS

    names = {t.__name__ for t in TOOLS}
    assert {"terrain_los", "terrain_horizon", "terrain_set_origin",
            "terrain_survey", "room_impulse", "sensor_query"} <= names


def test_cached_station_tile():
    """Real Terrarium tile 12/2018/1344 if the hoa64 cache has it."""
    from pathlib import Path

    cache = Path.home() / ".cache/hoa64/terrain/12/2018/1344.png"
    if not cache.exists():
        pytest.skip("station Terrarium tile not cached")
    geo.reset()
    origin = geo.load_origin()
    meta = geo.terrain_meta()
    assert origin["source"] == "terrarium"
    assert origin["tile_x"] == 2018
    assert meta["G"] == 255
    assert meta["hmax_m"] > meta["hmin_m"]
    geo.reset()


def test_survey_in_tile_cached():
    from pathlib import Path

    cache = Path.home() / ".cache/hoa64/terrain/12/2018/1344.png"
    if not cache.exists():
        pytest.skip("station Terrarium tile not cached")
    geo.reset()
    geo.load_origin()
    # both ends inside the resident tile
    res = geo.survey(52.447, -2.60, 52.444, -2.59, tx_h=15, rx_h=15, n=64)
    assert res["kind"] == "survey"
    assert res["path_m"] > 50
    assert res["multi_tile"] is False
    assert "verdict" in res
    assert res.get("rtu") is not None
    geo.reset()


def test_survey_multi_tile_cached():
    from pathlib import Path

    west = Path.home() / ".cache/hoa64/terrain/12/2017/1344.png"
    here = Path.home() / ".cache/hoa64/terrain/12/2018/1344.png"
    if not (west.exists() and here.exists()):
        pytest.skip("neighbor Terrarium tiles not cached")
    geo.reset()
    geo.load_origin()
    # ~4 km west of the station crosses into tile 2017
    res = geo.survey(52.445472, -2.597833, 52.445472, -2.655, tx_h=25, rx_h=25,
                     n=80)
    assert res["kind"] == "survey"
    assert res["path_m"] > 2000
    assert res["multi_tile"] is True
    assert len(res["dist_m"]) >= 8
    assert len(res["elev_m"]) == len(res["dist_m"])
    geo.reset()
