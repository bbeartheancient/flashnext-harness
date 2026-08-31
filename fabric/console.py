"""Task catalog — form schemas for the modular LCARS shell.

Each function is one left-rail button. The shell shows only that
function's I/O fields and the viewports it declares (`term`, `graph`,
or `both`). `when` on a field hides it unless another field matches.
"""

from __future__ import annotations

FUNCTIONS = [
    {"id": "chat", "title": "Chat", "group": "command",
     "method": "POST", "url": "/chat", "viewport": "term",
     "fields": [{"name": "message", "type": "text",
                 "value": "Computer, status"}]},
    {"id": "ship", "title": "Ship status", "group": "command",
     "method": "GET", "url": "/ship_status", "viewport": "both",
     "fields": []},
    {"id": "toks", "title": "tok/s", "group": "command",
     "method": "GET", "url": "/api/sensors/history?topic=ship/vllm/toks&seconds=900&key=val",
     "viewport": "both", "fields": [], "live": 5},
    {"id": "gpuwatts", "title": "GPU power", "group": "command",
     "method": "GET",
     "url": "/api/sensors/history?topic=ship/gpu/card1&seconds=900&key=power_w",
     "viewport": "both", "fields": [], "live": 5},
    {"id": "ports", "title": "Port status", "group": "command",
     "method": "GET", "url": "/api/ports", "viewport": "term",
     "fields": []},
    {"id": "goals", "title": "Goals", "group": "command",
     "method": "GET", "url": "/api/goals", "viewport": "term",
     "fields": []},
    {"id": "orbits", "title": "Solar system", "group": "navigation",
     "method": "GET", "url": "/api/lab/planets", "viewport": "both",
     "fields": [{"name": "date", "type": "text", "value": ""}]},
    {"id": "news", "title": "News feed", "group": "navigation",
     "method": "GET", "url": "/api/news", "viewport": "term",
     "fields": [{"name": "limit", "type": "number", "value": 20}]},
    {"id": "tickers", "title": "Stock tickers", "group": "operations",
     "method": "GET", "url": "/api/stocks?symbols=AAPL,MSFT,NVDA",
     "viewport": "term", "fields": [
         {"name": "symbols", "type": "text", "value": "AAPL,MSFT,NVDA"}]},
    {"id": "firefly", "title": "Finance (Firefly)", "group": "operations",
     "method": "GET", "url": "/api/firefly/net-worth", "viewport": "term",
     "fields": []},
    {"id": "omp", "title": "Coding agent (OMP)", "group": "command",
     "method": "POST", "url": "/api/omp/run", "viewport": "term",
     "fields": [{"name": "message", "type": "textarea", "value":
                 "Survey docs/ and summarize the harness in 5 bullets."}]},
    {"id": "netmap", "title": "LAN devices", "group": "security",
     "method": "GET", "url": "/api/net/devices", "viewport": "both",
     "fields": []},
    {"id": "mediastack", "title": "Media stack", "group": "communications",
     "method": "GET", "url": "/api/media/stack", "viewport": "term",
     "fields": []},
    {"id": "backlot", "title": "Backlot status", "group": "communications",
     "method": "GET", "url": "/api/backlot/status",
     "viewport": "term", "fields": []},
    {"id": "log", "title": "Log tail", "group": "command",
     "method": "GET", "url": "/log/recent", "viewport": "term",
     "fields": [{"name": "limit", "type": "number", "value": 20}]},
    {"id": "hadamard", "title": "Hadamard", "group": "science",
     "method": "GET", "url": "/api/lab/hadamard", "viewport": "both",
     "fields": [
         {"name": "n", "type": "number", "value": 8},
         {"name": "method", "type": "select", "value": "auto",
          "options": ["auto", "sylvester", "known"]},
     ]},
    {"id": "hoa_encode", "title": "HOA encode", "group": "science",
     "method": "POST", "url": "/api/lab/hoa", "viewport": "both",
     "fields": [
         {"name": "azimuths", "type": "text", "value": "90"},
         {"name": "elevations", "type": "text", "value": "0"},
         {"name": "order", "type": "number", "value": 4},
     ]},
    {"id": "hoa_decode", "title": "HOA decode", "group": "science",
     "method": "POST", "url": "/api/lab/hoa/decode", "viewport": "term",
     "fields": [
         {"name": "azimuths", "type": "text", "value": "90"},
         {"name": "elevations", "type": "text", "value": "0"},
         {"name": "order", "type": "number", "value": 4},
     ]},
    {"id": "hoa_rotate", "title": "HOA rotate", "group": "science",
     "method": "POST", "url": "/api/lab/hoa/rotate", "viewport": "both",
     "fields": [
         {"name": "azimuths", "type": "text", "value": "0"},
         {"name": "elevations", "type": "text", "value": "0"},
         {"name": "yaw_deg", "type": "number", "value": 90},
         {"name": "order", "type": "number", "value": 4},
     ]},
    {"id": "orbital", "title": "Orbital", "group": "science",
     "method": "GET", "url": "/api/lab/orbital", "viewport": "both",
     "fields": [
         {"name": "n", "type": "number", "value": 2},
         {"name": "l", "type": "number", "value": 1},
         {"name": "m", "type": "number", "value": 0},
     ]},
    {"id": "antenna", "title": "Antenna", "group": "science",
     "method": "GET", "url": "/api/lab/antenna", "viewport": "both",
     "fields": [
         {"name": "kind", "type": "select", "value": "yagi",
          "options": ["dipole", "monopole", "loop", "patch", "helix", "yagi"]},
         {"name": "f_mhz", "type": "number", "value": 145},
     ]},
    {"id": "filter", "title": "Filter", "group": "science",
     "method": "GET", "url": "/api/lab/filter", "viewport": "both",
     "fields": [
         {"name": "kind", "type": "select", "value": "lpf",
          "options": ["lpf", "hpf", "bpf", "bsf"]},
         {"name": "f_c_mhz", "type": "number", "value": 100},
         {"name": "n", "type": "number", "value": 5},
         {"name": "f_lo_mhz", "type": "number", "value": 90,
          "when": {"name": "kind", "in": ["bpf", "bsf"]}},
         {"name": "f_hi_mhz", "type": "number", "value": 110,
          "when": {"name": "kind", "in": ["bpf", "bsf"]}},
     ]},
    {"id": "link", "title": "Link budget", "group": "science",
     "method": "GET", "url": "/api/lab/link", "viewport": "term",
     "fields": [
         {"name": "p_tx_dbw", "type": "number", "value": 0},
         {"name": "g_tx_dbi", "type": "number", "value": 2.15},
         {"name": "g_rx_dbi", "type": "number", "value": 2.15},
         {"name": "f_mhz", "type": "number", "value": 5800},
         {"name": "d_m", "type": "number", "value": 100},
     ]},
    {"id": "survey", "title": "Terrain survey", "group": "science",
     "method": "POST", "url": "/api/geo/survey", "viewport": "both",
     "fields": [
         {"name": "tx_lat", "type": "number", "value": 52.445472},
         {"name": "tx_lon", "type": "number", "value": -2.597833},
         {"name": "rx_lat", "type": "number", "value": 52.445472},
         {"name": "rx_lon", "type": "number", "value": -2.655},
         {"name": "tx_h", "type": "number", "value": 25},
         {"name": "rx_h", "type": "number", "value": 25},
     ]},
    {"id": "los", "title": "LOS lat/lon", "group": "science",
     "method": "POST", "url": "/api/geo/los/latlon", "viewport": "term",
     "fields": [
         {"name": "tx_lat", "type": "number", "value": 52.445472},
         {"name": "tx_lon", "type": "number", "value": -2.597833},
         {"name": "rx_lat", "type": "number", "value": 52.445472},
         {"name": "rx_lon", "type": "number", "value": -2.655},
         {"name": "tx_h", "type": "number", "value": 25},
         {"name": "rx_h", "type": "number", "value": 25},
     ]},
    {"id": "place", "title": "Place", "group": "science",
     "method": "POST", "url": "/api/geo/place", "viewport": "both",
     "fields": [
         {"name": "q", "type": "text", "value": "Neosho, MO"},
         {"name": "lat", "type": "number"},
         {"name": "lon", "type": "number"},
         {"name": "zoom", "type": "number", "value": 12},
         {"name": "view", "type": "select", "value": "terrain",
          "options": ["terrain", "horizon"]},
     ]},
    {"id": "horizon", "title": "Horizon", "group": "science",
     "method": "POST", "url": "/api/geo/horizon", "viewport": "both",
     "fields": [
         {"name": "h_agl", "type": "number", "value": 25},
         {"name": "n_az", "type": "number", "value": 72},
         {"name": "elev_deg", "type": "number", "value": -2},
     ]},
    {"id": "room", "title": "Room IR", "group": "science",
     "method": "POST", "url": "/api/room/ir", "viewport": "both",
     "fields": [
         {"name": "src", "type": "text", "value": "2,1.5,2"},
         {"name": "lst", "type": "text", "value": "6,1.5,4"},
         {"name": "n_rays", "type": "number", "value": 256},
         {"name": "max_bounce", "type": "number", "value": 2},
     ]},
    {"id": "fdtd", "title": "FDTD", "group": "science",
     "method": "GET", "url": "/api/lab/fdtd", "viewport": "both",
     "fields": [
         {"name": "f_mhz", "type": "number", "value": 150},
         {"name": "medium", "type": "select", "value": "air",
          "options": ["air", "water"]},
         {"name": "n", "type": "number", "value": 16},
     ]},
    {"id": "materials", "title": "Materials", "group": "science",
     "method": "GET", "url": "/api/lab/materials", "viewport": "both",
     "fields": [
         {"name": "kind", "type": "select", "value": "cloth",
          "options": ["cloth", "touchpad", "metamaterial"]},
         {"name": "order", "type": "number", "value": 8},
     ]},
    {"id": "scales", "title": "Actual size", "group": "science",
     "method": "GET", "url": "/api/lab/scales", "viewport": "term",
     "fields": [{"name": "eps", "type": "number", "value": 0.003}]},
    {"id": "crown", "title": "Crown PSF", "group": "science",
     "method": "GET", "url": "/api/lab/crown", "viewport": "both",
     "fields": [{"name": "n", "type": "number", "value": 32}]},
    {"id": "sage", "title": "Sage / CAS", "group": "science",
     "method": "POST", "url": "/api/sage", "viewport": "term",
     "fields": [{"name": "expr", "type": "text",
                 "value": "factor(x**2-1)"}]},
    {"id": "eng_ship", "title": "Ship status", "group": "science",
     "method": "GET", "url": "/ship_status", "viewport": "both",
     "fields": []},
    {"id": "sensors", "title": "Sensor bus", "group": "holodeck",
     "method": "GET", "url": "/api/sensors", "viewport": "both",
     "fields": [{"name": "prefix", "type": "text", "value": "ship/"},
                {"name": "limit", "type": "number", "value": 60}]},
    {"id": "compose", "title": "New MiniApp", "group": "command",
     "method": "POST", "url": "/api/microapps/compose", "viewport": "both",
     "fields": [{"name": "query", "type": "text",
                 "value": "make a 7-day habit tracker"}]},
]


def catalog(group: str | None = None) -> dict:
    from . import departments

    fns = list(FUNCTIONS)
    try:
        from . import microapps

        fns.extend(microapps.list_public())
    except Exception:
        pass
    have = {f["group"] for f in fns}
    for d in departments.DEPARTMENTS:
        if d["id"] in have:
            continue
        fns.append({
            "id": f"stub-{d['id']}",
            "title": "Standby",
            "group": d["id"],
            "method": "GET",
            "url": f"/api/departments/{d['ext']}",
            "viewport": "term",
            "fields": [],
            "note": d.get("note") or "not wired",
        })
    if group == "engineering":
        group = "science"
    if group:
        fns = [f for f in fns if f["group"] == group]
    return {"group": group, "functions": fns}
