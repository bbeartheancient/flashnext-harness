"""Home Assistant actuator plane (D09, build order #3).

ha_call / ha_state against the LAN REST API. The HA server (RPi3B) is
currently disconnected — every function degrades to an honest error
until it is back. Token lives in fabric/data/media-keys.env
(FABRIC_HA_TOKEN) or the FABRIC_HA_TOKEN env var; never logged.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import secrets

_UA = "woodfire-fabric"


def _base() -> str:
    return os.environ.get(
        "FABRIC_HA_URL", "http://127.0.0.1:8123").rstrip("/")


def _token() -> str:
    env = os.environ.get("FABRIC_HA_TOKEN", "")
    if env.strip():
        return env.strip()
    return secrets.load().get("FABRIC_HA_TOKEN", "")


def _headers() -> dict:
    h = {"Content-Type": "application/json", "User-Agent": _UA}
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def up(timeout: float = 2.0) -> bool:
    """Cheap reachability probe; a 401 still means the server is up."""
    try:
        req = urllib.request.Request(_base() + "/api/",
                                     headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500
    except Exception:
        return False


def ha_state(entity_id: str) -> dict:
    """Read one entity state from Home Assistant."""
    entity_id = str(entity_id or "").strip()
    if not entity_id:
        return {"error": "missing entity_id"}
    if not up():
        return {"error": f"Home Assistant unreachable at {_base()} "
                         "(D09 offline)"}
    req = urllib.request.Request(_base() + f"/api/states/{entity_id}",
                                 headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HA HTTP {e.code} for {entity_id}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"entity_id": data.get("entity_id"),
            "state": data.get("state"),
            "attributes": data.get("attributes") or {},
            "friendly_name": (data.get("attributes") or {})
            .get("friendly_name")}


def ha_call(domain: str, service: str, entity_id: str = "",
            data: dict | None = None) -> dict:
    """Call a Home Assistant service.

    NON-IDEMPOTENT — confirm-gated per harness router rule 3 (lights,
    scenes, climate writes). domain/service like light/turn_on.
    """
    domain = str(domain or "").strip()
    service = str(service or "").strip()
    if not domain or not service:
        return {"error": "need domain and service (e.g. light, turn_on)"}
    if not up():
        return {"error": f"Home Assistant unreachable at {_base()} "
                         "(D09 offline)"}
    body: dict = {}
    if entity_id:
        body["entity_id"] = str(entity_id).strip()
    if isinstance(data, dict) and data:
        body.update(data)
    req = urllib.request.Request(
        _base() + f"/api/services/{domain}/{service}",
        data=json.dumps(body).encode(), headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode() or "[]")
    except urllib.error.HTTPError as e:
        return {"error": f"HA HTTP {e.code} on {domain}.{service}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    changed = [r.get("entity_id") for r in payload
               if isinstance(r, dict) and r.get("entity_id")]
    return {"ok": True, "domain": domain, "service": service,
            "changed": changed}
