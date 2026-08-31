"""latent.py: /v1/hidden_states contract + loop structure (mocked LLM)."""

import base64
import io
import json
import os
import selectors
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from afi.spectral import BLOCK

from fabric import latent as lat
from fabric.latent import _capture_hidden_states, _spectrum, latent_loop


@pytest.fixture(autouse=True)
def _no_gpu_worker(monkeypatch):
    """Fresh GPU lane per test, pointed at a nonexistent interpreter so
    the host-numpy fallback path is exercised deterministically."""
    monkeypatch.setattr(lat, "_GPU", lat._GpuFwht())
    monkeypatch.setenv("FABRIC_AFI_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("FABRIC_AFI_WORKER", "/nonexistent/worker.py")


def _hs_response(arr):
    return json.dumps({
        "hidden_fp16_le": base64.b64encode(
            arr.astype("<f2").tobytes()).decode(),
        "shape": list(arr.shape),
        "token_ids": [1] * arr.shape[0],
    }).encode()


class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_capture_sends_prompt_field():
    seen = {}
    arr = (np.random.default_rng(0)
           .standard_normal((4, 2560)).astype(np.float16))

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_hs_response(arr))

    with patch.object(lat.urllib.request, "urlopen", fake_urlopen):
        got, why = _capture_hidden_states("hello world")
    assert why == ""
    assert seen["url"].endswith("/v1/hidden_states")
    assert seen["body"]["prompt"] == "hello world"
    assert "text" not in seen["body"]
    assert seen["body"]["max_tokens"] == 2048
    assert got.shape == (4, 2560)
    assert got.dtype == np.float16


def test_capture_http_error_reason():
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":"missing prompt"}'))

    with patch.object(lat.urllib.request, "urlopen", fake_urlopen):
        got, why = _capture_hidden_states("hello")
    assert got is None
    assert why.startswith("HTTP 400:")
    assert "missing prompt" in why


def test_spectrum_cpu_parseval():
    rng = np.random.default_rng(7)
    # Qwen-FN: H=2560, sparse attn head_dim=BLOCK=256 → 10 blocks/token
    arr = rng.standard_normal((6, 2560)).astype(np.float16)
    spec = _spectrum(arr)
    assert spec.get("backend") == "cpu"
    nb = arr.shape[1] // BLOCK  # per-token block count
    expect = float((arr.astype(np.float64) ** 2).sum()
                   / (arr.shape[0] * nb))  # mean per-block energy
    assert abs(spec["mean_energy"] - expect) <= 0.02 * max(expect, 1.0)
    assert 0 < spec["compaction_90"] <= 1.0
    assert spec["top_bins"]


def test_spectrum_capture_failure_is_honest():
    spec = _spectrum(None, "HTTP 400: nope")
    assert spec == {"error": "HTTP 400: nope"}


def test_fingerprint_delta_is_flat():
    lat._ensure_afi_path()
    from afi.spectral import blocks_view, fingerprint

    # Qwen-FN hidden=2560; a delta in row 0 is FLAT in the orthonormal
    # Walsh basis (Parseval: energy is uniform across all 256 bins).
    delta = np.zeros((3, 2560), dtype=np.float32)
    delta[:, 0] = 1.0  # one-hot rows -> flat Walsh spectrum
    fp = fingerprint(blocks_view(delta))
    # orthonormal FWHT of a delta is flat: ~90% of coefficients needed
    assert fp["compaction_90"] > 0.85
    rng = np.random.default_rng(1)
    fp_rand = fingerprint(blocks_view(rng.standard_normal((8, 2560))))
    assert fp_rand["compaction_90"] < 0.8  # gaussian rows are compact


def test_loop_converges_first_iteration():
    arr = (np.random.default_rng(1)
           .standard_normal((2, 2560)).astype(np.float16) * 0.05)

    def mock_urlopen(req, timeout=0):
        if req.full_url.endswith("/hidden_states"):
            return _Resp(_hs_response(arr))
        return _Resp(json.dumps({"choices": [{"message":
                     {"content": "COMPLETE"}}]}).encode())

    with patch.object(lat.urllib.request, "urlopen", mock_urlopen):
        r = latent_loop("What is the answer?")
    assert r["converged"] is True
    assert r["iterations_run"] >= 1
    assert len(r["iterations"][0]["text"]) > 0
    assert r["iterations"][0]["spectrum"]["backend"] == "cpu"
    assert len(r["energy_trajectory"]) == 1


def test_loop_caps_at_max_iterations():
    arr = np.zeros((2, 2560), dtype=np.float16)

    def mock_urlopen(req, timeout=0):
        if req.full_url.endswith("/hidden_states"):
            return _Resp(_hs_response(arr))
        return _Resp(json.dumps({"choices": [{"message":
                     {"content": "needs more work"}}]}).encode())

    with patch.object(lat.urllib.request, "urlopen", mock_urlopen):
        r = latent_loop("do the thing", max_iterations=2)
    assert r["iterations_run"] == 2
    assert r["converged"] is False
    assert len(r["energy_trajectory"]) == 2


def test_convergence_tolerates_template_junk():
    """The duty GPTQ model wraps replies in chat-template debris; a bare
    'We need answer user: COMPLETE' verdict must still converge."""
    arr = np.zeros((1, 2560), dtype=np.float16)

    def mock_urlopen(req, timeout=0):
        if req.full_url.endswith("/hidden_states"):
            return _Resp(_hs_response(arr))
        content = "We need answer user: COMPLETE"
        return _Resp(json.dumps({"choices": [{"message":
                     {"content": content}}]}).encode())

    with patch.object(lat.urllib.request, "urlopen", mock_urlopen):
        r = latent_loop("x")
    assert r["converged"] is True


class _StubGPU:
    """Hermetic stand-in for the worker lane."""

    def __init__(self):
        self.compact_calls = []

    def fingerprint(self, arr):
        return {"backend": "stub", "mean_energy": 1.0,
                "compaction_90": 0.5, "rows": arr.shape[0] * (arr.shape[1] // BLOCK)}

    def compact(self, arr, rate):
        self.compact_calls.append((arr.shape, rate))
        return {"b64": "STUBPAYLOAD", "effective_rate": rate,
                "row_cos": 0.99}


def test_embeds_mode_on_refinement_legs(monkeypatch):
    monkeypatch.setenv("FABRIC_LATENT_COMPACT", "0.5")
    stub = _StubGPU()
    monkeypatch.setattr(lat, "_GPU", stub)
    seen = []
    arr = np.zeros((2, 2560), dtype=np.float16)

    def mock_urlopen(req, timeout=0):
        url = req.full_url
        body = json.loads(req.data.decode())
        if url.endswith("/hidden_states"):
            return _Resp(_hs_response(arr))
        seen.append(body)
        content = (body.get("messages") or [{}])[-1].get("content")
        if isinstance(content, list):  # embeds arm
            return _Resp(json.dumps({"choices":
                         [{"message": {"content":
                           "improved answer"}}]}).encode())
        if "Original request" in str(content):
            # first critique: needs work; second: done
            if not any(isinstance((s.get("messages") or [{}])[0].get("content"),
                                  list) for s in seen):
                return _Resp(json.dumps(
                    {"choices": [{"message":
                                  {"content": "add more detail"}}]}).encode())
            return _Resp(json.dumps({"choices": [{"message":
                         {"content": "COMPLETE"}}]}).encode())
        return _Resp(json.dumps({"choices": [{"message":
                     {"content": "first answer"}}]}).encode())

    with patch.object(lat.urllib.request, "urlopen", mock_urlopen):
        r = latent_loop("do the thing", max_iterations=2)
    assert r["iterations"][0]["gen_mode"] == "tokens"
    assert r["iterations"][1]["gen_mode"] == "embeds"
    assert r["iterations"][1]["compaction"]["requested_rate"] == 0.5
    assert len(stub.compact_calls) == 1
    embed_bodies = [b for b in seen
                    if isinstance((b.get("messages") or [{}])[0].get("content"),
                                  list)]
    assert len(embed_bodies) == 1
    parts = embed_bodies[0]["messages"][0]["content"]
    assert parts[0] == {"type": "prompt_embeds", "data": "STUBPAYLOAD"}
    assert parts[1]["type"] == "text"
    assert embed_bodies[0]["temperature"] == 0.0
    assert "model" in embed_bodies[0]


def test_embeds_failure_falls_back_to_tokens(monkeypatch):
    monkeypatch.setenv("FABRIC_LATENT_COMPACT", "0.5")

    class _DeadGPU:
        def fingerprint(self, arr):
            return None

        def compact(self, arr, rate):
            return None

    monkeypatch.setattr(lat, "_GPU", _DeadGPU())
    arr = np.zeros((1, 2560), dtype=np.float16)

    def mock_urlopen(req, timeout=0):
        if req.full_url.endswith("/hidden_states"):
            return _Resp(_hs_response(arr))
        body = json.loads(req.data.decode())
        assert "prompt_embeds" not in body
        if "messages" in body:
            content = body["messages"][-1]["content"]
            if "Original request" in content:
                return _Resp(json.dumps({"choices": [{"message":
                             {"content": "COMPLETE"}}]}).encode())
        return _Resp(json.dumps({"choices": [{"message":
                     {"content": "answer"}}]}).encode())

    with patch.object(lat.urllib.request, "urlopen", mock_urlopen):
        r = latent_loop("x", max_iterations=2)
    assert all(i["gen_mode"] == "tokens" for i in r["iterations"])


def test_gpu_lane_falls_back_after_failures():
    gpu = lat._GpuFwht()  # fixture env points at a nonexistent python
    arr = np.zeros((1, 128), dtype=np.float16)
    for _ in range(lat._GPU_MAX_FAILS):
        assert gpu.fingerprint(arr) is None
    assert gpu._fails >= lat._GPU_MAX_FAILS
    assert gpu.fingerprint(arr) is None  # lane disabled, no respawn try


def test_worker_live_b70(monkeypatch):
    py = (Path.home() / "runtime" / ".venv-vllm" / "bin" / "python")
    script = (Path(lat.__file__).resolve().parent.parent /
              "scripts" / "latent-fwht-worker.py")
    if not (script.is_file() and py.is_file()):
        pytest.skip("latent-fwht-worker or vLLM venv missing")
    monkeypatch.setenv("FABRIC_AFI_PYTHON", str(py))  # undo fixture stub
    monkeypatch.setenv("FABRIC_FWHT_BLOCK", str(BLOCK))  # Qwen-FN default
    proc = subprocess.Popen(
        [str(py), str(script)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rng = np.random.default_rng(3)
    # Qwen-FN: H=2560, BLOCK=256 → 10 blocks per token
    arr = rng.standard_normal((16, 2560)).astype(np.float16)
    body = json.dumps({
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": [16, 2560],
    }).encode() + b"\n"
    try:
        proc.stdin.write(body)
        proc.stdin.flush()
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        buf = bytearray()
        deadline = time.monotonic() + 180  # torch import can be slow
        while b"\n" not in buf:
            left = deadline - time.monotonic()
            assert left > 0, "worker timeout"
            for key, _mask in sel.select(left):
                chunk = os.read(key.fd, 65536)
                assert chunk, "worker died"
                buf.extend(chunk)
        sel.close()
        resp = json.loads(bytes(buf.split(b"\n")[0]))
        assert resp["ok"], resp.get("error")
        assert resp["backend"] in ("xpu", "cpu")
        expect = float((arr.astype(np.float64) ** 2).sum()
                       / (16 * (2560 // BLOCK)))  # Qwen-FN: H=2560, BLOCK=256
        got = resp["fp"]["mean_energy"]
        assert abs(got - expect) <= 0.05 * max(expect, 1.0)
    finally:
        proc.kill()
