"""XYZ vector lane: BPE state components as coordinates of the n-gram buffer.

User directive: "the BPE vectors can be used as the XYZ vectors in the n-gram
buffer". Each BPE component is one coordinate axis:

    Belief     -> X  (session "bpe-x")
    Progress   -> Y  (session "bpe-y")
    Experience -> Z  (session "bpe-z")

Write path per axis (bicameral framework §2 contract):

    rendered axis text
      -> hidden capture      (vLLM POST /hidden_states, GPU0 duty model)
      -> dct_topk gate       (row_cos >= NEOH_ROW_COS_GATE, default 0.95)
      -> buffer worker store (ple-buffer-worker JSON-line pipe, GPU1 lane,
                              mode="merge", persist=False for live turns)
      -> activate            (publishes the axis table as a replace-refresh)

Everything here is fail-soft and gated behind NEOH_LATENT=1: with the latent
lane off (the default) or down, the text harness keeps working unchanged.

Echo landmine (M1 AGENTS.md 2026-08-29): only harness/env-derived text is
ever stored — never the duty model's own replies.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import subprocess
import threading
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

AXES = ("x", "y", "z")
AXIS_SESSIONS = {"x": "bpe-x", "y": "bpe-y", "z": "bpe-z"}

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
ROW_COS_GATE = float(os.environ.get("NEOH_ROW_COS_GATE", "0.95"))
COMPACT_RATE = float(os.environ.get("NEOH_COMPACT_RATE", "0.5"))
TAP_LAYER = int(os.environ.get("NEOH_TAP_LAYER", "63"))
MAX_AXIS_CHARS = int(os.environ.get("NEOH_AXIS_CHARS", "8000"))
MAX_FAILS = 2
WORKER_TIMEOUT_S = 60
SPAWN_TIMEOUT_S = 300


def _atlas_default_rate() -> float | None:
    """Cache-map rate for the tap layer (atlas residency map as default).

    Returns None when the atlas/map is unavailable — callers fall back to
    COMPACT_RATE. Never raises: the latent lane must stay fail-soft.
    """
    try:
        from . import atlas as atlas_mod
        cmap = atlas_mod.cache_map(atlas_mod.load())
        return atlas_mod.rate_for(cmap, TAP_LAYER)
    except Exception:  # noqa: BLE001
        return None


def latent_enabled() -> bool:
    return os.environ.get("NEOH_LATENT", "0") == "1"


# ---------------------------------------------------------------------------
# dct_topk compaction gate (transport contract, bicameral framework §2)
#
# Local numpy implementation matching afi.mdct semantics (dct2 / keep_topk /
# idct2 / row_cos_mean) so the gate does not require the M1 tree. The M1
# `afi.mdct` module remains the correctness oracle.
# ---------------------------------------------------------------------------

def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    m[0] *= 1.0 / math.sqrt(2)
    return m * math.sqrt(2.0 / n)  # orthonormal DCT-II


def dct_topk_roundtrip(arr: np.ndarray, rate: float) -> tuple[np.ndarray, float]:
    """dct_topk compact at `rate` and reconstruct; returns (recon, row_cos)."""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {x.shape}")
    n = x.shape[1]
    dct = _dct_matrix(n)
    coeffs = x @ dct.T
    keep = max(1, int(round(n * rate)))
    # keep the largest-|coeff| columns per row (keep_topk)
    thresh = np.sort(np.abs(coeffs), axis=1)[:, n - keep][:, None]
    masked = np.where(np.abs(coeffs) >= thresh, coeffs, 0.0)
    recon = masked @ dct
    # row_cos_mean
    a = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    b = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)
    row_cos = float(np.mean(np.sum(a * b, axis=1)))
    return recon, row_cos


def passes_gate(arr: np.ndarray, rate: float | None = None,
                gate: float | None = None) -> tuple[bool, float]:
    """The bicameral transport gate: row_cos >= gate at the given rate."""
    if rate is None:
        rate = COMPACT_RATE
    if gate is None:
        gate = ROW_COS_GATE
    _, row_cos = dct_topk_roundtrip(arr, rate)
    return row_cos >= gate, row_cos


# ---------------------------------------------------------------------------
# Hidden capture (vLLM /hidden_states hook, GPU0)
# ---------------------------------------------------------------------------

def capture_hidden(text: str, base_url: str | None = None,
                   timeout: float = 600.0) -> np.ndarray:
    """POST {base}/hidden_states -> fp16 [L, H] array (hidden_fp16_le contract)."""
    base = (base_url or os.environ.get("NEOH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)).rstrip("/")
    body = json.dumps({"prompt": text, "max_tokens": 1}).encode()
    req = urllib.request.Request(
        base + "/hidden_states", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    shape = payload["shape"]
    raw = base64.b64decode(payload["hidden_fp16_le"])
    return np.frombuffer(raw, dtype="<f2").reshape(shape)


def arr_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f2").tobytes()).decode()


# ---------------------------------------------------------------------------
# Buffer worker client (ple-buffer-worker JSON-line pipe, GPU1 lane)
# ---------------------------------------------------------------------------

class BufferWorker:
    """Speaks the M1 ple-buffer-worker JSON-line protocol over stdin/stdout.

    Protocol (verbatim from M1 scripts/ple-buffer-worker.py):
      ready:  {"ok": true, "ready": true, ...}
      store:  {"op":"store","text","b64","shape":[L,H],"session",
               "mode":"merge"|"replace","persist":bool}
      activate: {"op":"activate","session"}
      stats:  {"op":"stats"}
    """

    def __init__(self, cmd: list[str] | None = None,
                 spawn_timeout: float = SPAWN_TIMEOUT_S,
                 request_timeout: float = WORKER_TIMEOUT_S):
        if cmd is None:
            python = os.environ.get(
                "NEOH_PLE_BUFFER_PYTHON",
                str(Path.home() / "runtime/.venv/bin/python"),
            )
            worker = os.environ.get(
                "NEOH_PLE_BUFFER_WORKER",
                str(Path.home() / "M1Multitronic/scripts/ple-buffer-worker.py"),
            )
            cmd = [python, worker]
        self.cmd = cmd
        self.spawn_timeout = spawn_timeout
        self.request_timeout = request_timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._fails = 0

    @property
    def dead(self) -> bool:
        return self._fails >= MAX_FAILS

    def _spawn(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = os.environ.copy()
        # Preprocessor card (GPU1), same ruling as M1 fabric's _PleBuffer:
        # the worker's XPU GEMM must not land on vLLM's GPU0.
        env["ZE_AFFINITY_MASK"] = os.environ.get("NEOH_PLE_BUFFER_GPU", "1")
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
        )
        line = self._readline(self.spawn_timeout)
        msg = json.loads(line)
        if not (msg.get("ok") and msg.get("ready")):
            raise RuntimeError(f"buffer worker not ready: {msg}")

    def _readline(self, timeout: float) -> str:
        import selectors

        assert self._proc and self._proc.stdout
        sel = selectors.DefaultSelector()
        sel.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            if not sel.select(timeout):
                raise TimeoutError(f"buffer worker timeout after {timeout}s")
            line = self._proc.stdout.readline()
        finally:
            sel.close()
        if not line:
            raise RuntimeError("buffer worker closed stdout")
        return line

    def _ask(self, body: dict) -> dict | None:
        if self.dead:
            return None
        with self._lock:
            try:
                self._spawn()
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(json.dumps(body) + "\n")
                self._proc.stdin.flush()
                resp = json.loads(self._readline(self.request_timeout))
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error", "unknown worker error"))
                self._fails = 0
                return resp
            except Exception as exc:
                self._fails += 1
                log.warning("buffer worker op %s failed (%d/%d): %s",
                            body.get("op"), self._fails, MAX_FAILS, exc)
                if self._proc is not None:
                    self._proc.kill()
                    self._proc = None
                return None

    def store(self, session: str, text: str, arr: np.ndarray,
              mode: str = "merge", persist: bool = False) -> dict | None:
        return self._ask({
            "op": "store", "text": text, "b64": arr_to_b64(arr),
            "shape": list(arr.shape), "session": session,
            "mode": mode, "persist": persist,
        })

    def activate(self, session: str) -> dict | None:
        return self._ask({"op": "activate", "session": session})

    def stats(self) -> dict | None:
        return self._ask({"op": "stats"})


# ---------------------------------------------------------------------------
# XYZ buffer: the three coordinate axes
# ---------------------------------------------------------------------------

class XYZBuffer:
    """Publishes BPE axis state into the n-gram buffer as XYZ coordinates."""

    def __init__(self, worker: BufferWorker | None = None,
                 base_url: str | None = None,
                 enabled: bool | None = None,
                 rate: float | None = None):
        self.worker = worker  # lazily constructed if None and enabled
        self.base_url = base_url
        self.enabled = latent_enabled() if enabled is None else enabled
        # rate resolution: explicit arg > atlas cache map (tap layer) > global
        self.rate = rate if rate is not None else (
            _atlas_default_rate() if self.enabled else None)
        self.gate_rejections: list[dict] = []

    def _get_worker(self) -> BufferWorker | None:
        if self.worker is None:
            self.worker = BufferWorker()
        return self.worker

    def store_axis(self, axis: str, text: str, persist: bool = False,
                   mode: str = "merge") -> dict | None:
        """Mirror one axis's rendered text into the buffer as latent edges.

        Returns the worker's store response, or None when the lane is
        disabled/down or the row_cos gate rejects the capture.
        """
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
        if not self.enabled:
            return None
        session = AXIS_SESSIONS[axis]
        text = text[:MAX_AXIS_CHARS]
        try:
            arr = capture_hidden(text, self.base_url)
        except Exception as exc:
            log.warning("hidden capture failed for axis %s: %s", axis, exc)
            return None
        ok, row_cos = passes_gate(arr, rate=self.rate)
        if not ok:
            self.gate_rejections.append({"axis": axis, "row_cos": row_cos})
            log.warning("axis %s rejected by transport gate: row_cos=%.4f < %.2f",
                        axis, row_cos, ROW_COS_GATE)
            return None
        worker = self._get_worker()
        resp = worker.store(session, text, arr, mode=mode, persist=persist)
        if resp is None:
            return None
        worker.activate(session)
        resp["row_cos"] = row_cos
        resp["axis"] = axis
        resp["session"] = session
        return resp

    def stats(self) -> dict | None:
        if not self.enabled:
            return None
        return self._get_worker().stats()
