"""Per-department pane registries (harness-plan catalog).

Departments are contexts, not menus: each bundles a pane set, a default
deck layout, and a 3-digit phone extension. Global Command-strip panes
(chat / ship / log) stay reachable from every department. Unwired lanes
still appear so the rail is the ABI; their panes are stubs.
"""

from __future__ import annotations

DEPARTMENTS: list[dict] = [
    {
        "ext": 100,
        "id": "command",
        "name": "Command",
        "scope": "operator line: chat, status, log, tracker, mode pin",
        "panes": ["chat", "ship", "log", "tracker", "cmd", "sensors"],
        "default_slots": ["chat", "ship", "log"],
        "layout": "side",
        "wired": True,
        "color": "a9",
    },
    {
        "ext": 200,
        "id": "flight",
        "name": "Navigation",
        "scope": "AV/sensor rig: GoStream, PTZ, glass seating, LED matrix, Quest",
        "panes": ["chat", "ship", "flight"],
        "default_slots": ["chat", "flight", "ship"],
        "layout": "side",
        "wired": False,
        "note": "D01–D04 video lane not wired",
        "color": "a2",
    },
    {
        "ext": 400,
        "id": "science",
        "name": "Sciences",
        "scope": "hoa64 lab + ship systems: terrain, HOA/Hadamard, room-IR, RF, MQTT sensors, power caps",
        "panes": ["console", "geo", "lab", "room", "sensors", "ship", "chat"],
        "default_slots": ["console", "geo", "lab"],
        "layout": "triple",
        "wired": True,
        "aliases": [300],
        "note": "ext 300 (Engineering) aliases here. HA / laser not wired; Mosquitto live.",
        "color": "a4",
    },
    {
        "ext": 500,
        "id": "medical",
        "name": "Medical",
        "scope": "crew maintenance, hygiene reports, habit/med timers",
        "panes": ["chat", "log", "medical"],
        "default_slots": ["chat", "medical", "log"],
        "layout": "side",
        "wired": False,
        "note": "task_log / timers not wired",
        "color": "a3",
    },
    {
        "ext": 600,
        "id": "security",
        "name": "Security",
        "scope": "CCTV frames/events, IP whitelist, guardian, alerts",
        "panes": ["chat", "ship", "security"],
        "default_slots": ["chat", "security", "ship"],
        "layout": "side",
        "wired": False,
        "note": "D05 CCTV not wired",
        "color": "ex1",
    },
    {
        "ext": 700,
        "id": "operations",
        "name": "Operations",
        "scope": "library/downloads, asset lifecycle, finance",
        "panes": ["chat", "log", "operations"],
        "default_slots": ["chat", "operations", "log"],
        "layout": "side",
        "wired": False,
        "note": "Jellyfin merge not wired; D15 parked",
        "color": "a6",
    },
    {
        "ext": 800,
        "id": "communications",
        "name": "Media",
        "scope": "phone system, news globe, outbound notices",
        "panes": ["chat", "log", "communications"],
        "default_slots": ["chat", "communications", "log"],
        "layout": "side",
        "wired": False,
        "note": "D07 phone / D16 globe not wired",
        "color": "a5",
    },
    {
        "ext": 900,
        "id": "holodeck",
        "name": "Holodeck",
        "scope": "Backlot-mode entry, studio programs, Deck state graph",
        "panes": ["chat", "log", "holodeck"],
        "default_slots": ["chat", "holodeck", "log"],
        "layout": "side",
        "wired": False,
        "note": "Track 3 job spine not merged",
        "color": "a8",
    },
]

GLOBAL_PANES = ("chat", "ship", "log")
# Retired Engineering (300) pins the merged Sciences department.
ALIASES = {300: 400}


def catalog() -> dict:
    return {
        "global_panes": list(GLOBAL_PANES),
        "departments": list(DEPARTMENTS),
        "aliases": dict(ALIASES),
    }


def by_ext(ext: int) -> dict | None:
    ext = ALIASES.get(int(ext), int(ext))
    for d in DEPARTMENTS:
        if d["ext"] == ext:
            return dict(d)
    return None


def pane_order(ext: int) -> list[str]:
    dept = by_ext(ext) or by_ext(100)
    seen = []
    for p in list(dept["panes"]) + list(GLOBAL_PANES) + ["blank"]:
        if p not in seen:
            seen.append(p)
    return seen
