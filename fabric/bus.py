"""Sensor bus — MQTT-shaped topics, sqlite record, optional broker.

Harness-plan #2. Mosquitto is local on 127.0.0.1:1883 (pacman 2.1.2).
The record of truth is a sqlite topic log (`sensors` in fabric.db);
when FABRIC_MQTT_HOST is set and paho is importable, publishes are also
mirrored to the broker. `sensor_query(prefix, since)` is the agent tool.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from .log_store import get_log_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sensors_ts ON sensors(ts DESC);
CREATE INDEX IF NOT EXISTS idx_sensors_topic ON sensors(topic);
"""

_lock = threading.Lock()
_pub_thread = None
_pub_stop = threading.Event()


_db = {"c": None, "path": None}


def _dbconn():
    path = get_log_store().path
    if _db["c"] is None or _db["path"] != path:
        c = sqlite3.connect(path, check_same_thread=False)
        c.executescript(_SCHEMA)
        _db["c"] = c
        _db["path"] = path
    return _db["c"]


def publish(topic: str, payload, source: str = "fabric") -> dict:
    """Append one reading. payload is JSON-able."""
    ts = time.time()
    body = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    topic = str(topic)
    with _lock:
        _dbconn().execute(
            "INSERT INTO sensors (ts, topic, payload) VALUES (?, ?, ?)",
            (ts, topic, body),
        )
        _dbconn().commit()
    _maybe_mqtt(topic, body)
    return {"ts": ts, "topic": topic, "source": source}


def query(prefix: str = "ship/", since: float | None = None,
          limit: int = 50) -> list[dict]:
    """Recent readings whose topic starts with `prefix`, newer than `since`."""
    limit = int(max(1, min(int(limit), 500)))
    sql = "SELECT id, ts, topic, payload FROM sensors WHERE topic LIKE ?"
    params: list = [str(prefix) + "%"]
    if since is not None:
        sql += " AND ts >= ?"
        params.append(float(since))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _lock:
        rows = _dbconn().execute(sql, params).fetchall()
    out = []
    for i, ts, topic, payload in rows:
        try:
            val = json.loads(payload)
        except Exception:
            val = payload
        out.append({"id": i, "ts": ts, "topic": topic, "payload": val})
    return out


def publish_gpus() -> list[dict]:
    from . import status_sources

    rows = []
    for g in status_sources.gpu_hwmon():
        card = g.get("card") or "unknown"
        rows.append(publish(f"ship/gpu/{card}", g, source="hwmon"))
    return rows


_VLLM_LAST: dict | None = None


def publish_vllm() -> list[dict]:
    """tok/s + MTP acceptance scraped from duty vLLM /metrics."""
    global _VLLM_LAST
    import urllib.request

    base = os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    url = base.rsplit("/v1", 1)[0] + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            text = resp.read().decode()
    except Exception:
        return []
    gen = acc = dra = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total{"):
            try:
                gen += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            try:
                acc += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            try:
                dra += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    now = time.time()
    out = []
    if _VLLM_LAST and gen >= _VLLM_LAST["total"]:
        dt = now - _VLLM_LAST["ts"]
        rate = (gen - _VLLM_LAST["total"]) / dt if dt > 0 else 0.0
        accept = ((acc - _VLLM_LAST.get("acc", 0.0))
                  / (dra - _VLLM_LAST.get("dra", 0.0))
                  if dra > _VLLM_LAST.get("dra", 0.0) else None)
        out.append(publish("ship/vllm/toks",
                           {"val": round(rate, 1),
                            "accept": round(accept, 2)
                            if accept is not None else None},
                           source="vllm"))
    _VLLM_LAST = {"ts": now, "total": gen, "acc": acc, "dra": dra}
    return out


def _mqtt_host() -> str | None:
    return os.environ.get("FABRIC_MQTT_HOST", "127.0.0.1")


def _maybe_mqtt(topic: str, payload: str) -> None:
    host = _mqtt_host()
    if not host:
        return
    try:
        import paho.mqtt.publish as mqtt_pub  # noqa: WPS433
    except ImportError:
        return
    try:
        mqtt_pub.single(
            topic, payload, hostname=host,
            port=int(os.environ.get("FABRIC_MQTT_PORT", "1883")),
            keepalive=5,
        )
    except Exception:
        return


def start_publisher(interval_s: float = 5.0) -> None:
    global _pub_thread
    if _pub_thread and _pub_thread.is_alive():
        return
    _pub_stop.clear()

    def loop():
        while not _pub_stop.is_set():
            try:
                publish_gpus()
            except Exception:
                pass
            try:
                publish_vllm()
            except Exception:
                pass
            _pub_stop.wait(interval_s)

    _pub_thread = threading.Thread(target=loop, name="fabric-bus", daemon=True)
    _pub_thread.start()


def stop_publisher() -> None:
    _pub_stop.set()


def broker_status() -> dict:
    import socket

    host = _mqtt_host()
    listening = False
    if host:
        try:
            with socket.socket() as s:
                s.settimeout(0.2)
                listening = s.connect_ex((host, int(os.environ.get("FABRIC_MQTT_PORT", "1883")))) == 0
        except OSError:
            listening = False
    return {
        "record": "sqlite",
        "mqtt_host": host,
        "mqtt_mirror": bool(host) and listening,
        "mqtt_listening": listening,
        "publisher_alive": bool(_pub_thread and _pub_thread.is_alive()),
    }
