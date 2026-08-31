"""Media stack tools — Jellyfin + *arr on the NAS host (build order #4).

Ported from OlympusServer backlot/services/media_service.py, trimmed to
what the harness needs: reachability, library search, play handoff,
queue counts. Zero-dep urllib; every call degrades to an honest error
dict when the media server is off (it sleeps).
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

from . import secrets

_UA = "woodfire-fabric"

_PORTS = {
    "jellyfin": 8096,
    "sonarr": 8989,
    "radarr": 7878,
    "prowlarr": 9696,
    "bazarr": 6767,
    "qbittorrent": 8080,
    "jellyseerr": 5055,
}

_KEY_NAMES = {
    "jellyfin": "BACKLOT_JELLYFIN_API_KEY",
    "sonarr": "BACKLOT_SONARR_API_KEY",
    "radarr": "BACKLOT_RADARR_API_KEY",
    "qbittorrent": "BACKLOT_QBIT_APIKEY",
}


def _host() -> str:
    return os.environ.get("FABRIC_MEDIA_HOST", "127.0.0.1")


def _key(service: str) -> str:
    env = os.environ.get(_KEY_NAMES.get(service, ""), "")
    if env.strip():
        return env.strip()
    return secrets.load().get(_KEY_NAMES[service], "")


def _url(service: str, path: str) -> str:
    return f"http://{_host()}:{_PORTS[service]}{path}"


def _get(url: str, timeout: float = 4.0,
         headers: dict | None = None,
         retries: int = 1) -> tuple[int | None, object]:
    """GET with JSON decode; one retry rides out NAS spin-up stalls."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                               **(headers or {})})
    result: tuple[int | None, object] = (None, {"error": "no attempt"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
        except urllib.error.HTTPError as e:
            return e.code, {"error": f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001
            result = (None, {"error": str(e)})
            if attempt < retries:
                time.sleep(0.6)
    return result


def _post(url: str, payload: dict, timeout: float = 6.0,
          headers: dict | None = None) -> tuple[int | None, object]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"User-Agent": _UA, "Content-Type": "application/json",
                 **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, {"error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return None, {"error": str(e)}
    if not body:
        return 200, {}
    try:
        return 200, json.loads(body)
    except json.JSONDecodeError:
        return 200, {}


def _port_up(port: int) -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(1.2)
            return s.connect_ex((_host(), port)) == 0
    except OSError:
        return False


def stack_status() -> dict:
    """TCP reachability for every registered service + Jellyfin identity."""
    rows = {}
    for svc, port in sorted(_PORTS.items()):
        rows[svc] = _port_up(port)
    out: dict = {"host": _host(), "services": rows}
    if rows["jellyfin"]:
        code, info = _get(_url("jellyfin", "/System/Info/Public"))
        if isinstance(info, dict) and code == 200:
            out["jellyfin_version"] = info.get("Version")
            out["server_name"] = info.get("ServerName")
    elif not any(rows.values()):
        out["note"] = "media host unreachable (asleep or off LAN)"
    return out


def jellyfin_stats() -> dict:
    """Jellyfin /Items/Counts (library size). Needs the Jellyfin key."""
    key = _key("jellyfin")
    if not key:
        return {"error": "no Jellyfin API key (media-keys.env)"}
    code, data = _get(_url("jellyfin", "/Items/Counts"),
                      headers={"X-Emby-Token": key})
    if code != 200 or not isinstance(data, dict):
        err = data.get("error") if isinstance(data, dict) else "bad reply"
        return {"error": f"jellyfin stats failed: {err}"}
    return data


def jellyfin_search(query: str, limit: int = 10) -> dict:
    """Search the Jellyfin library (movies/series/episodes/music)."""
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    key = _key("jellyfin")
    if not key:
        return {"error": "no Jellyfin API key (media-keys.env)"}
    from urllib.parse import quote
    path = ("/Items?Recursive=true"
            "&IncludeItemTypes=Movie,Series,Episode,MusicAlbum,Audio"
            f"&Limit={max(1, min(int(limit), 25))}"
            f"&Fields=ProductionYear&searchTerm={quote(q)}")
    code, data = _get(_url("jellyfin", path),
                      headers={"X-Emby-Token": key}, timeout=12.0)
    if code != 200 or not isinstance(data, dict):
        err = data.get("error") if isinstance(data, dict) else "bad reply"
        return {"error": f"jellyfin search failed: {err}"}
    items = []
    for it in data.get("Items") or []:
        items.append({
            "id": it.get("Id"),
            "name": it.get("Name"),
            "type": it.get("Type"),
            "year": it.get("ProductionYear"),
        })
    return {"query": q, "total": data.get("TotalRecordCount"), "items": items}


def jellyfin_play(item_id: str, client: str = "") -> dict:
    """Start playback of a Jellyfin item on an active session.

    NON-IDEMPOTENT — confirm-gated per router rule 3. client matches
    Client or DeviceName; empty picks the only active session.
    """
    item_id = str(item_id or "").strip()
    if not item_id:
        return {"error": "missing item_id"}
    key = _key("jellyfin")
    if not key:
        return {"error": "no Jellyfin API key (media-keys.env)"}
    hdrs = {"X-Emby-Token": key}
    code, sessions = _get(_url("jellyfin", "/Sessions"), headers=hdrs)
    if code != 200 or not isinstance(sessions, list):
        return {"error": "could not list sessions (server down?)"}
    live = [s for s in sessions if isinstance(s, dict)]
    target = None
    if client:
        low = client.lower()
        target = next((s for s in live
                       if low in str(s.get("Client", "")).lower()
                       or low in str(s.get("DeviceName", "")).lower()), None)
        if target is None:
            return {"error": f"no active session matching '{client}'"}
    else:
        if len(live) == 1:
            target = live[0]
        elif len(live) > 1:
            return {"error": "multiple active sessions; pass client=",
                    "sessions": [f"{s.get('Client')}:{s.get('DeviceName')}"
                                 for s in live]}
        else:
            return {"error": "no active playback session"}
    sid = target.get("Id")
    code, resp = _post(_url("jellyfin", f"/Sessions/{sid}/Playing"),
                       {"ItemIds": [item_id], "PlayCommand": "PlayNow"},
                       headers=hdrs)
    if code != 200:
        err = resp.get("error") if isinstance(resp, dict) else "failed"
        return {"error": f"play command failed: {err}"}
    return {"ok": True, "session": target.get("DeviceName"),
            "item_id": item_id}


def arr_queue() -> dict:
    """Sonarr + Radarr download-queue counts (needs their API keys)."""
    out: dict = {}
    for svc, label in (("sonarr", "sonarr"), ("radarr", "radarr")):
        key = _key(svc)
        if not key:
            out[label] = {"error": f"no {_KEY_NAMES[svc]}"}
            continue
        code, data = _get(_url(svc, "/api/v3/queue?pagesize=1"),
                          headers={"X-Api-Key": key})
        if code != 200 or not isinstance(data, dict):
            out[label] = {"error": "unreachable" if code is None
                          else f"HTTP {code}"}
        else:
            out[label] = {"queued": data.get("totalRecords")}
    return out
