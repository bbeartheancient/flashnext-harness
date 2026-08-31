"""backlot.py job-spine client vs a fake :8001."""

import json
from unittest.mock import patch

from fabric import backlot


def _serve(rows):
    seq = list(rows)

    def _open(req, timeout=0):
        code, payload = seq.pop(0)
        if isinstance(code, Exception):
            raise code

        class R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode()

        return R()

    return _open


def test_status_parses_health(monkeypatch):
    monkeypatch.delenv("FABRIC_BACKLOT_API_KEY", raising=False)
    open_ = _serve([(200, {
        "status": "ok",
        "vllm": {"ok": True, "status": 200},
        "comfyui": {"ok": True,
                    "data": {"system": {"comfyui_version": "0.33.1"}}},
    })])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.backlot_status()
    assert out["ok"] is True
    assert out["vllm_ok"] is True
    assert out["comfyui_version"] == "0.33.1"


def test_submit_rejects_unknown_program():
    out = backlot.submit("rm_rf_slash")
    assert "error" in out and "program must be" in out["error"]


def test_submit_returns_job_id(monkeypatch):
    open_ = _serve([(200, {"job_id": 42, "status": "pending"})])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.submit("image", "a lighthouse in fog")
    assert out["ok"] is True and out["job_id"] == 42
    assert out["status"] == "pending"


def test_job_status_unwraps_result_paths(monkeypatch):
    open_ = _serve([(200, {
        "id": 7, "workflow_name": "zimage_base", "status": "done",
        "result_paths": '["/out/a.png"]', "error_message": None,
    })])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.job_status(7)
    assert out["status"] == "done"
    assert out["result_paths"] == ["/out/a.png"]
    assert out["error"] is None


def test_job_status_bad_id():
    assert "error" in backlot.job_status("abc")


def test_train_submit_requires_name():
    assert "error" in backlot.train_submit({})
    assert "error" in backlot.train_submit("nope")


def test_train_jobs_lists(monkeypatch):
    open_ = _serve([(200, [
        {"id": 3, "name": "h3-lora", "status": "running",
         "target_arch": "wan"},
        {"id": 2, "name": "ace-test", "status": "done",
         "target_arch": "ace_step"},
    ])])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.train_jobs()
    assert len(out["jobs"]) == 2
    assert out["jobs"][0]["name"] == "h3-lora"


def test_down_is_honest(monkeypatch):
    def refused(req, timeout=0):
        raise OSError("refused")

    with patch.object(backlot.urllib.request, "urlopen", refused):
        out = backlot.backlot_status()
    assert "unreachable" in out["error"]
