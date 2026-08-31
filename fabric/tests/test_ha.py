"""ha.py: honest offline behaviour + correct service-call payloads."""

import json
from unittest.mock import patch

from fabric import ha


def test_offline_is_honest():
    with patch.object(ha, "up", return_value=False):
        s = ha.ha_state("light.desk")
        c = ha.ha_call("light", "turn_on", "light.desk")
    assert "unreachable" in s["error"]
    assert "unreachable" in c["error"]
    assert "D09" in s["error"]


def test_call_payload_shape(monkeypatch):
    seen = {}

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                [{"entity_id": "light.desk", "state": "on"}]).encode()

    def fake_open(req, timeout=0):
        if getattr(req, "data", None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            seen["auth"] = req.get_header("Authorization")
        return R()

    monkeypatch.setattr(ha.urllib.request, "urlopen", fake_open)
    monkeypatch.setenv("FABRIC_HA_TOKEN", "tok123")
    out = ha.ha_call("light", "turn_on", "light.desk",
                     {"brightness": 128})
    assert out.get("ok") is True
    assert seen["url"].endswith("/api/services/light/turn_on")
    assert seen["body"] == {"entity_id": "light.desk",
                            "brightness": 128}
    assert seen["auth"] == "Bearer tok123"


def test_state_parses_entity(monkeypatch):
    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "entity_id": "sensor.temp", "state": "21.5",
                "attributes": {"friendly_name": "Desk Temp"}}).encode()

    def fake_open(req, timeout=0):
        return R()

    monkeypatch.setattr(ha.urllib.request, "urlopen", fake_open)
    monkeypatch.setenv("FABRIC_HA_TOKEN", "t")
    out = ha.ha_state("sensor.temp")
    assert out["state"] == "21.5"
    assert out["friendly_name"] == "Desk Temp"


def test_up_false_on_conn_refused(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("refused")

    monkeypatch.setattr(ha.urllib.request, "urlopen", boom)
    assert ha.up() is False


def test_missing_args_rejected():
    assert "error" in ha.ha_state("")
    assert "error" in ha.ha_call("", "")
