"""media.py against a fake Jellyfin/*arr host."""

import json
from unittest.mock import patch

from fabric import media


def _serve(rows):
    """rows: list of (status_or_None, payload) consumed per request."""
    seq = list(rows)

    def _open(req, timeout=0):
        code, payload = seq.pop(0)
        body = json.dumps(payload).encode()

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return body

            status = 200 if code == 200 else 500
        return R()

    return _open


def test_stack_status_all_down(monkeypatch):
    monkeypatch.delenv("FABRIC_MEDIA_HOST", raising=False)
    monkeypatch.setattr(media, "_port_up", lambda port: False)
    out = media.stack_status()
    assert out["host"] == "127.0.0.1"
    assert not any(out["services"].values())
    assert "note" in out


def test_jellyfin_search_parses_items(monkeypatch):
    monkeypatch.setenv("FABRIC_MEDIA_HOST", "127.0.0.1")
    open_ = _serve([(200, {"TotalRecordCount": 2, "Items": [
        {"Id": "a1", "Name": "Dune", "Type": "Movie",
         "ProductionYear": 1984},
        {"Id": "b2", "Name": "Dune (TV)", "Type": "Series",
         "ProductionYear": 2000}]})])
    with patch.object(media.urllib.request, "urlopen", open_), \
            patch.object(media.secrets, "load",
                         return_value={"BACKLOT_JELLYFIN_API_KEY": "K"}):
        out = media.jellyfin_search("dune")
    assert out["total"] == 2 and len(out["items"]) == 2
    assert out["items"][0]["id"] == "a1"
    assert out["items"][0]["year"] == 1984


def test_play_needs_session(monkeypatch):
    monkeypatch.setenv("FABRIC_MEDIA_HOST", "127.0.0.1")
    open_ = _serve([(200, [])])
    with patch.object(media.urllib.request, "urlopen", open_), \
            patch.object(media.secrets, "load",
                         return_value={"BACKLOT_JELLYFIN_API_KEY": "K"}):
        out = media.jellyfin_play("x1")
    assert "error" in out and "no active" in out["error"]


def test_play_targets_named_client(monkeypatch):
    monkeypatch.setenv("FABRIC_MEDIA_HOST", "127.0.0.1")
    open_ = _serve([
        (200, [{"Id": "s1", "Client": "Jellyfin Web",
                "DeviceName": "couch"},
               {"Id": "s2", "Client": "Roku", "DeviceName": "den"}]),
        (200, {}),
    ])
    bodies = []
    orig_post = media._post

    def spy_post(url, payload, timeout=6.0, headers=None):
        bodies.append((url, payload))
        return orig_post(url, payload, timeout=timeout,
                         headers=headers)

    with patch.object(media.urllib.request, "urlopen", open_), \
            patch.object(media, "_post", spy_post), \
            patch.object(media.secrets, "load",
                         return_value={"BACKLOT_JELLYFIN_API_KEY": "K"}):
        out = media.jellyfin_play("x1", client="den")
    assert out.get("ok") is True
    assert "/Sessions/s2/Playing" in bodies[0][0]
    assert bodies[0][1]["ItemIds"] == ["x1"]


def test_arr_queue_reports_missing_keys():
    with patch.object(media.secrets, "load", return_value={}):
        out = media.arr_queue()
    assert "error" in out["sonarr"] and "error" in out["radarr"]
