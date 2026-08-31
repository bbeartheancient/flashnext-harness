"""Looping transformer — iterative refinement with spectral tracking.

NOT a DiT. Not weight modification. A structured self-refinement loop:
  1. Generate from prompt
  2. Capture final hidden states via POST /v1/hidden_states
     (vllm_hidden_hook contract: {"prompt": <raw string>} ->
      {"hidden_fp16_le": <b64 [L,H] fp16 LE>, "shape": [L, H]})
  3. Rotate through torch.ops.afi.fwht on the B70
     (scripts/latent-fwht-worker.py; host-numpy fallback) -> spectral
     fingerprint (energy + compaction) via afi/spectral.py
  4. Self-critique -> augment context for next iteration
  5. Repeat until model signals COMPLETE or max iterations

Each iteration's text + fingerprint persists to an .mv2 shard: the
latent state file system (time-indexed spectral frames per iteration).
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .log_store import get_log_store

HIDDEN_TIMEOUT_S = 600.0  # /v1/hidden_states round trip; Qwen-FN prefill is slow on cold start (MoE routing)
WORKER_TIMEOUT_S = 60.0   # per-request pipe deadline
SPAWN_TIMEOUT_S = 300.0   # worker cold start (torch xpu init + tokenizer)
MAX_PROMPT_TOKENS = 2048  # endpoint hard cap (vllm_hidden_hook)
CAPTURE_CHARS = 4000      # keep the capture well under the token cap
_GPU_MAX_FAILS = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_afi_path() -> None:
    p = str(_repo_root() / "python")
    if p not in sys.path:
        sys.path.insert(0, p)


def _llm_url() -> str:
    return os.environ.get("FABRIC_LLM_BASE_URL",
                          "http://127.0.0.1:8000/v1").rstrip("/")


def arr_to_b64(arr: np.ndarray) -> str:
    """base64 of the contiguous little-endian fp16 bytes of ``arr``
    (the worker-pipe and prompt_embeds wire encoding)."""
    return base64.b64encode(np.ascontiguousarray(
        arr, dtype="<f2").tobytes()).decode("ascii")


def _model_id() -> str:
    from .agent import discover_model

    return discover_model()


def _chat(messages: list[dict], timeout: float = 120) -> str:
    from . import harness

    return harness.post_llm(
        _llm_url(), os.environ.get("FABRIC_LLM_API_KEY", "local"),
        _model_id(), messages, orch="local", timeout=timeout)


class _GpuFwht:
    """Persistent latent-fwht-worker subprocess (torch.ops.afi.fwht).

    The fabric venv has no torch, so the rotation runs in the vLLM venv
    helper over a JSON-line pipe. After repeated failures the GPU lane
    is disabled for the process lifetime and the host-numpy fallback
    takes over."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = None
        self._fails = 0

    def _python(self) -> str:
        return os.environ.get(
            "FABRIC_AFI_PYTHON",
            str(Path.home() / "runtime" / ".venv-vllm" / "bin" / "python"))

    def _script(self) -> Path:
        env = os.environ.get("FABRIC_AFI_WORKER")
        if env:
            return Path(env)
        return _repo_root() / "scripts" / "latent-fwht-worker.py"

    def _spawn(self) -> None:
        py, script = self._python(), self._script()
        if not (Path(py).is_file() and script.is_file()):
            raise RuntimeError(f"worker missing ({py} / {script})")
        # BICAMERAL distribution (operator ruling 2026-08-26):
        #   vLLM  lives on GPU0 (x16 B70, Gen5 x16)  — the active card.
        #   Worker lives on GPU1 (x4  B70, Gen4 x4)  — the preprocessor.
        # FABRIC_AFI_GPU defaults to "1" (the slower card, by design).
        # Override via the env var if the topology ever flips.
        env = os.environ.copy()
        gpu = os.environ.get("FABRIC_AFI_GPU", "1")
        env["ZE_AFFINITY_MASK"] = gpu
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        env["NEO_ReadDeviceBinaryBuiltins"] = "0"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["CCL_TOPO_P2P_ACCESS"] = "0"
        self._proc = subprocess.Popen(
            [py, str(script)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None

    def close(self) -> None:
        with self._lock:
            self._kill()

    def compact(self, arr: np.ndarray, rate: float) -> dict | None:
        """dct_topk-compact [L, H] along the token axis; returns
        {"b64": torch.save'd fp16 [L,H], "effective_rate": ...} for the
        /v1/completions prompt_embeds field, or None if unavailable.
        Shares the GPU lane's failure budget with fingerprint()."""
        with self._lock:
            if self._fails >= _GPU_MAX_FAILS:
                return None
            try:
                resp = self._ask(arr, op="compact", rate=rate)
                self._fails = 0
                return {"b64": resp["b64"],
                        "effective_rate": resp.get("effective_rate"),
                        "row_cos": resp.get("row_cos"),
                        "rows": resp.get("rows")}
            except Exception:  # noqa: BLE001
                self._kill()
                self._fails += 1
                return None

    def compact_b64(self, b64: str, shape, rate: float) -> dict | None:
        """compact() for an already-encoded fp16 payload — decodes to
        [L, H] and reuses the same pipe lane (avoids the caller having
        to hold the numpy array)."""
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype="<f2")
            arr = arr.reshape(int(shape[0]), int(shape[1]))
        except Exception:  # noqa: BLE001
            return None
        return self.compact(arr, rate)

    def _ask(self, arr: np.ndarray, op: str = "spectra",
             rate: float | None = None) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        body: dict = {"op": op,
                      "shape": [int(arr.shape[0]), int(arr.shape[1])]}
        if rate is not None:
            body["rate"] = float(rate)
        body["b64"] = arr_to_b64(arr)
        self._proc.stdin.write((json.dumps(body) + "\n").encode())
        self._proc.stdin.flush()
        buf = bytearray()
        sel = selectors.DefaultSelector()
        sel.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            deadline = time.monotonic() + WORKER_TIMEOUT_S
            while b"\n" not in buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("latent-fwht-worker timeout")
                for key, _mask in sel.select(left):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        raise RuntimeError("latent-fwht-worker died")
                    buf.extend(chunk)
        finally:
            sel.close()
        resp = json.loads(bytes(buf.split(b"\n", 1)[0]))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "worker error"))
        return resp

    def fingerprint(self, arr: np.ndarray) -> dict | None:
        """Spectral fingerprint of [L, H] rows, or None if unavailable."""
        with self._lock:
            if self._fails >= _GPU_MAX_FAILS:
                return None
            try:
                fp = self._ask(arr).get("fp") or {}
                self._fails = 0
                return fp
            except Exception:  # noqa: BLE001
                self._kill()
                self._fails += 1
                return None


_GPU = _GpuFwht()
atexit.register(_GPU.close)


class _PleBuffer:
    """Persistent ple-buffer-worker subprocess (writable n-gram edges).

    GPU1 service: takes a captured hidden span + its text, computes the
    160-dim edges on XPU, upserts latest-wins, write-behinds to the mv2
    shard, and publishes ple_refresh.{npz,seq} to the vLLM spool so the
    engine hot-swaps the edge table before the next forward step.
    Same failure-budget discipline as _GpuFwht: after _GPU_MAX_FAILS
    the lane is disabled for the process lifetime (Arm B falls back to
    plain token refinement)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = None
        self._fails = 0

    def _python(self) -> str:
        return os.environ.get(
            "FABRIC_PLE_BUFFER_PYTHON",
            str(Path.home() / "runtime" / ".venv" / "bin" / "python"))

    def _script(self) -> Path:
        env = os.environ.get("FABRIC_PLE_BUFFER_WORKER")
        if env:
            return Path(env)
        return _repo_root() / "scripts" / "ple-buffer-worker.py"

    def _spawn(self) -> None:
        py, script = self._python(), self._script()
        if not (Path(py).is_file() and script.is_file()):
            raise RuntimeError(f"ple-buffer worker missing ({py} / {script})")
        env = os.environ.copy()
        # Preprocessor card (operator ruling 2026-08-26), same as _GpuFwht.
        env["ZE_AFFINITY_MASK"] = os.environ.get("FABRIC_PLE_BUFFER_GPU", "1")
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        self._proc = subprocess.Popen(
            [py, str(script)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None

    def close(self) -> None:
        with self._lock:
            self._kill()

    def _readline(self, timeout: float) -> bytes:
        buf = bytearray()
        sel = selectors.DefaultSelector()
        sel.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            deadline = time.monotonic() + timeout
            while b"\n" not in buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("ple-buffer-worker timeout")
                for key, _mask in sel.select(left):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        raise RuntimeError("ple-buffer-worker died")
                    buf.extend(chunk)
        finally:
            sel.close()
        return bytes(buf.split(b"\n", 1)[0])

    def _ask(self, body: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
            # Cold start (torch xpu init + tokenizer) is way over the
            # per-request deadline; wait for the readiness line first.
            ready = json.loads(self._readline(SPAWN_TIMEOUT_S))
            if not ready.get("ready"):
                raise RuntimeError(f"ple-buffer-worker init: {ready}")
        self._proc.stdin.write((json.dumps(body) + "\n").encode())
        self._proc.stdin.flush()
        resp = json.loads(self._readline(WORKER_TIMEOUT_S))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "worker error"))
        return resp

    def store(self, text: str, arr: np.ndarray,
              mode: str = "replace", session: str = "default",
              persist: bool = False) -> dict | None:
        """Store one captured span; returns worker stats or None."""
        with self._lock:
            if self._fails >= _GPU_MAX_FAILS:
                return None
            try:
                resp = self._ask({
                    "op": "store", "text": text, "mode": mode,
                    "session": session, "persist": bool(persist),
                    "shape": [int(arr.shape[0]), int(arr.shape[1])],
                    "b64": base64.b64encode(np.ascontiguousarray(
                        arr, dtype="<f2").tobytes()).decode("ascii"),
                })
                self._fails = 0
                return {k: resp.get(k) for k in
                        ("seq", "session", "unique_ids", "n_new",
                         "evicted", "pinned", "table_rows",
                         "gemm_ms", "mv2_ms", "mv2_error", "note")
                        if resp.get(k) is not None}
            except Exception:  # noqa: BLE001
                self._kill()
                self._fails += 1
                return None

    def activate(self, session: str) -> dict | None:
        """Publish ``session``'s table as the engine refresh (replace)."""
        with self._lock:
            if self._fails >= _GPU_MAX_FAILS:
                return None
            try:
                resp = self._ask({"op": "activate", "session": session})
                self._fails = 0
                return {k: resp.get(k) for k in ("seq", "session",
                                                 "table_rows")
                        if resp.get(k) is not None}
            except Exception:  # noqa: BLE001
                self._kill()
                self._fails += 1
                return None


_PLE_BUFFER = _PleBuffer()
atexit.register(_PLE_BUFFER.close)


class EmbedBlockStore:
    """Rolling store of compacted prompt-embeds blocks, one per session
    (newest-wins).  Blocks live on disk so they survive fabric restarts;
    a TTL reaps stale ones.  The opencode shim splices the active block
    into post-compaction requests as a prompt_embeds message."""

    def __init__(self, dir_: Path | None = None) -> None:
        self._dir = dir_ or Path(os.environ.get(
            "FABRIC_EMBED_BLOCK_DIR",
            str(Path.home() / ".local" / "state" / "fabric"
                / "embed-blocks")))
        self._lock = threading.Lock()

    def _path(self, session: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in session)
        return self._dir / f"{safe}.json"

    def save(self, session: str, b64: str, rows: int,
             meta: dict | None = None) -> dict:
        rec = {"session": session, "b64": b64, "rows": int(rows),
               "ts": time.time()}
        if meta:
            rec.update(meta)
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(session).with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(self._path(session))
        return {k: rec[k] for k in ("session", "rows", "ts")}

    def active(self, session: str,
               ttl_h: float | None = None) -> dict | None:
        """The session's current block, or None if absent/expired.
        Expired blocks are reaped on read."""
        with self._lock:
            p = self._path(session)
            try:
                rec = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                return None
            ttl = ttl_h if ttl_h is not None else _embed_ttl_h()
            if ttl > 0 and time.time() - float(rec.get("ts", 0)) > ttl * 3600:
                try:
                    p.unlink()
                except OSError:
                    pass
                return None
            return rec

    def clear(self, session: str) -> bool:
        with self._lock:
            try:
                self._path(session).unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError:
                return False


EMBED_BLOCKS = EmbedBlockStore()


def _embed_enabled() -> bool:
    """FABRIC_OPENCODE_EMBED=1: the opencode shim's embed leg — on a
    detected compaction the pre-compaction conversation is compacted
    into a prompt_embeds block (this store), and subsequent turns carry
    it as a soft prompt before the last user message."""
    return os.environ.get("FABRIC_OPENCODE_EMBED") == "1"


def _embed_rate() -> float:
    try:
        return float(os.environ.get("FABRIC_OPENCODE_EMBED_RATE", "0.5")
                     or 0.5)
    except ValueError:
        return 0.5


def _embed_budget() -> int:
    return int(os.environ.get("FABRIC_OPENCODE_EMBED_CHARS", "48000"))


def _embed_gate() -> float:
    """Minimum compact row_cos for a block to be accepted (0 = accept
    whatever the worker returns)."""
    try:
        return float(os.environ.get("FABRIC_OPENCODE_EMBED_GATE", "0")
                     or 0)
    except ValueError:
        return 0.0


def _embed_max_rows() -> int:
    return int(os.environ.get("FABRIC_OPENCODE_EMBED_MAX_ROWS", "2048"))


def _embed_ttl_h() -> float:
    try:
        return float(os.environ.get("FABRIC_OPENCODE_EMBED_TTL_H", "24")
                     or 24)
    except ValueError:
        return 24.0


def _ple_buffer_enabled() -> bool:
    """Arm B gate: FABRIC_PLE_BUFFER=1 routes each iteration's captured
    span through the PLE edge buffer and the follow-up leg becomes a
    suffix-only plain chat (the refreshed edge table carries the span
    delta instead of token context)."""
    return os.environ.get("FABRIC_PLE_BUFFER") == "1"


def ngram_transfer(text: str, mode: str = "merge",
                   max_chars: int | None = None,
                   session: str = "default") -> dict | None:
    """Session-compaction transfer: turn a text span into n-gram edges.

    Captures final-layer hidden for ``text`` on the live engine, then
    hands the span to ple-buffer-worker (XPU edge GEMM → latest-wins
    upsert → mv2 write-behind → ple_refresh publish).  The engine's
    execute_model poll applies the refreshed table before the NEXT
    forward, so the post-compaction chat carries the elided context as
    a layer-63 residual delta instead of token history.

    ``mode="merge"`` accumulates successive transfers on top of the
    trained prior (the worker seeds from edges_layer63.npz);
    ``"replace"`` makes the span the sole delta source.  Unlike the
    live per-turn feed (``live_note``), this path persists to the mv2
    shard — it is the archive lane.

    Returns worker stats, or None on any failure (strictly
    best-effort — compaction must never fail because the lane is down).
    """
    text = (text or "").strip()
    if not text:
        return None
    budget = max_chars or int(os.environ.get(
        "FABRIC_NGRAM_COMPACT_CHARS", "16000"))
    arr, why = _capture_hidden_states(text, max_chars=budget)
    if arr is None:
        return {"error": f"capture failed: {why}"}
    note = _PLE_BUFFER.store(text[:budget], arr, mode=mode,
                             session=session, persist=True)
    if note is None:
        return {"error": "lane down"}
    return note


def _ngram_live_enabled() -> bool:
    """FABRIC_NGRAM_LIVE=1: per-turn rolling injection feed (the
    session-scoped KV-analog buffer; supersedes compaction-only
    transfers for turn-level recency)."""
    return os.environ.get("FABRIC_NGRAM_LIVE") == "1"


def live_session() -> str:
    return os.environ.get("FABRIC_NGRAM_LIVE_SESSION", "chat")


def live_note(session: str, text: str,
              max_chars: int | None = None) -> dict | None:
    """Live per-turn feed: capture ``text`` and merge it into
    ``session``'s rolling edge buffer (ephemeral — no mv2 write).
    Best-effort; returns worker stats or an error dict."""
    text = (text or "").strip()
    if not text:
        return None
    budget = max_chars or int(os.environ.get(
        "FABRIC_NGRAM_LIVE_CHARS", "8000"))
    arr, why = _capture_hidden_states(text, max_chars=budget)
    if arr is None:
        return {"error": f"capture failed: {why}"}
    note = _PLE_BUFFER.store(text[:budget], arr, mode="merge",
                             session=session, persist=False)
    if note is None:
        return {"error": "lane down"}
    return note


def live_note_arr(session: str, text: str, arr: np.ndarray,
                  max_chars: int | None = None) -> dict | None:
    """live_note with a pre-captured hidden matrix — used when the
    caller already captured the span (shared-capture paths) so the
    engine isn't hit twice."""
    text = (text or "").strip()
    if not text:
        return None
    budget = max_chars or int(os.environ.get(
        "FABRIC_NGRAM_LIVE_CHARS", "8000"))
    note = _PLE_BUFFER.store(text[:budget], arr, mode="merge",
                             session=session, persist=False)
    if note is None:
        return {"error": "lane down"}
    return note


def embed_transfer(session: str, span: str,
                   budget: int | None = None,
                   arr: np.ndarray | None = None) -> dict | None:
    """Compact ``span`` into a prompt_embeds block for ``session``.

    Captures final-layer hidden (unless ``arr`` is supplied — shared
    capture), dct_topk-compacts it on the GPU lane, and saves the
    block to EMBED_BLOCKS (newest-wins).  Gate: FABRIC_OPENCODE_EMBED_
    GATE (min row_cos) and _MAX_ROWS.  Best-effort; returns the save
    record or an error dict, None when the leg is disabled."""
    if not _embed_enabled():
        return None
    budget = budget or _embed_budget()
    span = (span or "")[:budget]
    if arr is None:
        if not span.strip():
            return None
        arr, why = _capture_hidden_states(span, max_chars=budget)
        if arr is None:
            return {"error": f"capture failed: {why}"}
    payload = _GPU.compact_b64(arr_to_b64(arr), arr.shape, _embed_rate())
    if payload is None:
        return {"error": "lane down"}
    rows = int(payload.get("rows") or 0)
    if rows <= 0 or rows > _embed_max_rows():
        return {"error": f"bad rows {rows}"}
    rcos = payload.get("row_cos")
    gate = _embed_gate()
    if rcos is not None and float(rcos) < gate:
        return {"error": f"row_cos {rcos} below gate {gate}"}
    return EMBED_BLOCKS.save(session, payload["b64"], rows,
                             {"effective_rate":
                              payload.get("effective_rate"),
                              "row_cos": rcos})


def embed_transfer_capture(session: str, arr: np.ndarray) -> dict | None:
    """embed_transfer from an already-captured hidden matrix (shared
    capture with the n-gram leg when the budgets match)."""
    return embed_transfer(session, "", arr=arr)


def _capture_hidden_states(text: str,
                           max_chars: int = CAPTURE_CHARS
                           ) -> tuple[np.ndarray | None, str]:
    """POST /v1/hidden_states (vllm_hidden_hook contract, PR 7).

    The prompt string goes in verbatim under ``prompt`` — the chat
    template is never applied server-side; the reply carries base64
    fp16 final-hidden rows. Returns ([L, H] matrix, "") on success or
    (None, reason) on any failure. ``max_chars`` bounds the prompt; the
    endpoint's own token cap (65536) is the hard limit.
    """
    body = json.dumps({"prompt": text[:max_chars],
                       "max_tokens": MAX_PROMPT_TOKENS}).encode()
    req = urllib.request.Request(
        _llm_url() + "/hidden_states", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "woodfire-fabric"})
    try:
        with urllib.request.urlopen(req, timeout=HIDDEN_TIMEOUT_S) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = str(json.loads(e.read().decode()).get("error", ""))
        except Exception:  # noqa: BLE001
            detail = e.reason
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    try:
        shape = payload["shape"]
        arr = np.frombuffer(base64.b64decode(payload["hidden_fp16_le"]),
                            dtype="<f2")
        arr = arr.reshape(int(shape[0]), int(shape[1]))
    except Exception as e:  # noqa: BLE001
        return None, f"bad hidden payload: {type(e).__name__}: {e}"
    return np.asarray(arr), ""


def _spectrum(arr: np.ndarray | None, why: str = "") -> dict:
    """Backend-tagged spectral fingerprint. Best effort, honest errors:
    XPU worker first, host-numpy FWHT fallback, else an error note."""
    if arr is None:
        return {"error": why or "capture failed"}
    fp = _GPU.fingerprint(arr)
    if fp is not None:
        return {"backend": "xpu", **fp}
    _ensure_afi_path()
    try:
        from afi.fwht import fwht as host_fwht
        from afi.spectral import blocks_view, fingerprint

        rows = blocks_view(arr.astype(np.float32))
        coeffs = host_fwht(rows, normalize=True).astype(np.float32)
        return {"backend": "cpu", **fingerprint(coeffs)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"spectral analysis failed: "
                         f"{type(e).__name__}: {e}"}


def _compact_rate() -> float:
    """Context-compaction keep-rate (Phase 3). 0 disables embeds mode;
    e.g. FABRIC_LATENT_COMPACT=0.5 reinjects prior context as a
    dct_topk-compacted soft prompt via /v1/completions."""
    try:
        return float(os.environ.get("FABRIC_LATENT_COMPACT", "0") or 0)
    except ValueError:
        return 0.0


def _render_transcript(messages: list[dict], extra: str = "") -> str:
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: "
             f"{m['content']}" for m in messages]
    if extra:
        lines.append(extra)
    return "\n\n".join(lines)


def _build_embeds(transcript: str, rate: float) -> dict | None:
    """Capture + compact the transcript into a prompt_embeds payload.
    Returns {"b64", "effective_rate", "row_cos"} or None on any failure
    (embeds mode is strictly best-effort; token chat is the fallback)."""
    arr, why = _capture_hidden_states(transcript)
    if arr is None:
        return None
    payload = _GPU.compact(arr, rate)
    if payload is None:
        return None
    return payload


def _chat_embeds(embeds_b64: str, suffix: str = "",
                 max_tokens: int = 1024) -> str:
    """POST /v1/chat/completions with a prompt_embeds content part.

    The wire format is base64 of a torch.save'd (L, H) CPU fp16 tensor
    (serialized by latent-fwht-worker.py — the fabric venv has no
    torch). Server-side, the embeds sentinel expands to placeholder
    token IDs (so MTP4's token-id bookkeeping stays valid) whose input
    vectors are replaced by the tensor rows; a text suffix stays real
    tokens."""
    parts: list[dict] = [{"type": "prompt_embeds", "data": embeds_b64}]
    if suffix:
        parts.append({"type": "text", "text": suffix})
    body = json.dumps({
        "model": _model_id(),
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        _llm_url() + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "woodfire-fabric"})
    with urllib.request.urlopen(req, timeout=HIDDEN_TIMEOUT_S) as r:
        payload = json.loads(r.read().decode())
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("empty chat completions response")
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def latent_loop(prompt: str, max_iterations: int = 3,
                session_tag: str = "") -> dict:
    """Iterative refinement loop with a spectral fingerprint per
    iteration (the .mv2 latent state file system records one frame per
    iteration). With FABRIC_LATENT_COMPACT set (e.g. 0.5), refinement
    legs carry the conversation as a dct_topk-compacted soft prompt
    instead of growing token history."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "empty prompt"}

    messages = [{"role": "user", "content": prompt}]
    iterations: list[dict] = []
    trajectory: list[float] = []
    converged = False
    compact_rate = _compact_rate()
    ple_buffer = _PLE_BUFFER if _ple_buffer_enabled() else None

    for it in range(max_iterations):
        t0 = time.time()
        gen_mode = "tokens"
        compaction_note: dict | None = None
        text = None
        if it > 0 and compact_rate > 0.0 and ple_buffer is None:
            # Embeds carry the bulky prior-turn context; the original
            # request + critique stay REAL tokens (the A/B shows pure
            # soft-prompt input without a text anchor degenerates).
            transcript = _render_transcript(messages)
            payload = _build_embeds(transcript[:CAPTURE_CHARS],
                                    compact_rate)
            if payload is not None:
                try:
                    improvement = messages[-1]["content"]
                    text = _chat_embeds(
                        payload["b64"],
                        suffix=(f"Original request: {prompt}\n"
                                f"Improvement needed: {improvement}\n"
                                "Provide the improved answer."))
                    gen_mode = "embeds"
                    compaction_note = {
                        "requested_rate": compact_rate,
                        "effective_rate": payload.get("effective_rate"),
                        "row_cos": payload.get("row_cos"),
                    }
                except Exception as e:  # noqa: BLE001
                    compaction_note = {"error":
                                       f"{type(e).__name__}: {e}"}
        if text is None:
            try:
                text = _chat(messages)
            except Exception as e:
                return {"error": f"generation failed at iteration {it}: {e}",
                        "iterations": iterations}
        elapsed = round(time.time() - t0, 2)

        # hidden-state capture + spectral analysis (best effort)
        arr, why = _capture_hidden_states(text)
        spectrum = _spectrum(arr, why)
        energy = spectrum.get("mean_energy")
        compaction_90 = spectrum.get("compaction_90")
        if isinstance(energy, (int, float)):
            trajectory.append(round(float(energy), 4))
        # BICAMERAL §6.6: publish the spectral compaction signal to
        # the MicroFX buffer bus so any EEL2 patch can read it as
        # input('compaction_90').  Best-effort; the latent loop must
        # not fail if the buffer module is unavailable.
        if isinstance(compaction_90, (int, float)):
            try:
                from .qwen_buffer import publish_compaction
                publish_compaction(float(compaction_90))
            except Exception:  # noqa: BLE001
                pass

        # self-critique: ask the model if the answer is complete
        critique_prompt = (
            f"Original request: {prompt}\n\n"
            f"Your answer:\n{text}\n\n"
            "Is this answer complete and correct? Reply with exactly "
            "'COMPLETE' if nothing more is needed, otherwise reply with "
            "one specific improvement to make."
        )
        try:
            critique = _chat([
                {"role": "user", "content": critique_prompt},
            ], timeout=60)
        except Exception:  # noqa: BLE001
            critique = "COMPLETE"

        # Verdict judged on the reply's first line after dropping the duty
        # GPTQ model's chat-template debris ("We need answer user: ...").
        words = re.findall(r"[A-Z]+", critique.splitlines()[0].upper()
                           if critique.strip() else "")
        tail = [w for w in words
                if w not in {"WE", "NEED", "RESPOND", "TO", "USER",
                             "ANSWER"}]
        converged = tail == ["COMPLETE"]
        # A/B harness gate: force the full iteration count regardless of
        # the self-critique verdict (the duty model self-approves almost
        # every answer, which starves multi-iteration measurements).
        if os.environ.get("FABRIC_LATENT_FORCE_ITERS"):
            converged = False

        entry = {
            "iteration": it,
            "text": text[:2000],
            "elapsed_s": elapsed,
            "converged": converged,
            "critique": critique[:500],
            "spectrum": spectrum,
            "gen_mode": gen_mode,
        }
        if compaction_note is not None:
            entry["compaction"] = compaction_note
        iterations.append(entry)

        # persist one latent frame to the mv2 shard
        try:
            from .dept_memory import add as dm_add

            dm_add("Sciences", json.dumps({
                "tag": session_tag or "latent_loop",
                "iteration": it,
                "converged": converged,
                "backend": spectrum.get("backend"),
                "mean_energy": spectrum.get("mean_energy"),
                "compaction_90": spectrum.get("compaction_90"),
                "gen_mode": gen_mode,
                "text_len": len(text),
            }), title=f"latent-{session_tag}-it{it}")
        except Exception:  # noqa: BLE001
            pass
        # BICAMERAL §6.6: fire the mv2_tap trigger so any EEL2 patch
        # listening on the bus sees the iteration boundary.
        try:
            from .qwen_buffer import fire_mv2_tap
            fire_mv2_tap(f"{session_tag}-it{it}")
        except Exception:  # noqa: BLE001
            pass

        if converged:
            break

        # Arm B: publish this iteration's captured span to the PLE edge
        # buffer (replace mode — the buffer, not the trained prior, is
        # the sole source of delta).  The worker writes ple_refresh.seq
        # before responding; the engine's execute_model poll applies it
        # before the follow-up's prefill forward (maintenance-window
        # feature: requires the engine-side ple.py apply_refresh patch).
        buffer_note: dict | None = None
        if ple_buffer is not None and arr is not None:
            # arr was captured from text[:CAPTURE_CHARS]; store the same
            # string so worker tokenization matches the row count.
            buffer_note = ple_buffer.store(text[:CAPTURE_CHARS], arr,
                                           mode="replace")
            entry["ple_buffer"] = buffer_note or {"error": "lane down"}

        # augment context with the critique for next iteration
        if ple_buffer is not None and buffer_note:
            # Suffix-only prompt: the span delta rides the refreshed
            # edge table, so the follow-up drops the token history.
            messages = [{"role": "user", "content": (
                f"Original request: {prompt}\n"
                f"Improvement needed: {critique}\n"
                "Provide the improved answer.")}]
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             f"Improvement needed: {critique}\n"
                             "Provide the improved answer."})

    final_text = iterations[-1]["text"] if iterations else ""
    return {
        "ok": True,
        "prompt": prompt[:200],
        "iterations_run": len(iterations),
        "converged": converged,
        "final_text": final_text[:2000],
        "iterations": iterations,
        "energy_trajectory": trajectory,
        "total_time_s": round(sum(i["elapsed_s"] for i in iterations), 2),
    }
