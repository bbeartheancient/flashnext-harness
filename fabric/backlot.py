"""Backlot job-spine client (build order #5).

Backlot stays its own app + webUI (backlot.woodfireindustries.com →
:8001). Fabric is the orchestrating CLIENT: submit studio/training
jobs, poll durable status, never run long work inline (router rule 4).
Backlot's API already returns {job_id, "pending"} immediately and its
workers own ComfyUI/GPU arbitration.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from . import secrets

_UA = "woodfire-fabric"

# Studio programs backed by durable GenerationJob workers. Inline-LLM
# routes (script/pitch/text) are deliberately NOT here — those block.
_PROGRAMS = ("image", "sphere", "mesh", "upscale", "sample", "sfx")


def _base() -> str:
    return os.environ.get(
        "FABRIC_BACKLOT_URL", "http://127.0.0.1:8001").rstrip("/")


def _key() -> str:
    env = os.environ.get("FABRIC_BACKLOT_API_KEY", "")
    if env.strip():
        return env.strip()
    return secrets.load().get("BACKLOT_API_KEY", "")


def _headers(extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json", "User-Agent": _UA}
    key = _key()
    if key:
        h["X-API-Key"] = key
    if extra:
        h.update(extra)
    return h


def _request(method: str, path: str, payload: dict | None = None,
             timeout: float = 8.0) -> dict:
    req = urllib.request.Request(
        _base() + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        return {"error": f"backlot HTTP {e.code}: {detail or e.reason}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"backlot unreachable at {_base()}: {e}"}
    if not body:
        return {}
    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else {"data": data}
    except json.JSONDecodeError:
        return {"error": "backlot returned non-JSON"}


def up() -> bool:
    out = _request("GET", "/system/health", timeout=4)
    return not out.get("error")


def backlot_status() -> dict:
    """Backlot service health + worker stack (vLLM/ComfyUI flags)."""
    out = _request("GET", "/system/health", timeout=5)
    if out.get("error"):
        return out
    vllm = out.get("vllm") or {}
    comfy = out.get("comfyui") or {}
    return {
        "ok": out.get("status") == "ok",
        "url": _base(),
        "vllm_ok": bool(vllm.get("ok")),
        "comfyui_ok": bool(comfy.get("ok")),
        "comfyui_version": ((comfy.get("data") or {}).get("system") or {})
        .get("comfyui_version"),
    }


def programs() -> dict:
    """Studio program templates available for backlot_submit."""
    out = _request("GET", "/generate/workflows")
    rows_src = out.get("workflows") if isinstance(out.get("workflows"), list) \
        else (out.get("data") if isinstance(out.get("data"), list) else [])
    rows = []
    for row in rows_src:
        if isinstance(row, str):
            rows.append(row)
        elif isinstance(row, dict):
            n = row.get("name") or row.get("id") or ""
            t = row.get("title") or ""
            if n:
                rows.append(f"{n} — {t}" if t else n)
    return {
        "submitable": list(_PROGRAMS),
        "workflows": rows[:40],
    }


def submit(program: str, prompt: str = "",
           params: dict | None = None) -> dict:
    """Queue a studio generation job; returns {job_id, status}.

    NON-IDEMPOTENT — confirm with the operator first. program is one of
    image/sphere/mesh/upscale/sample/sfx; extra fields ride params.
    """
    program = str(program or "").strip().lower()
    if program not in _PROGRAMS:
        return {"error": f"program must be one of {sorted(_PROGRAMS)}"}
    body: dict = {}
    if prompt:
        body["prompt"] = str(prompt)
    if isinstance(params, dict):
        body.update(params)
    out = _request("POST", f"/generate/{program}", payload=body,
                   timeout=15)
    if out.get("error"):
        return out
    return {"ok": True, "program": program,
            "job_id": out.get("job_id"),
            "status": out.get("status", "pending"),
            "note": f"poll with backlot_job_status({out.get('job_id')})"}


def job_status(job_id: int) -> dict:
    """Durable status of one Backlot generation job."""
    try:
        jid = int(job_id)
    except (TypeError, ValueError):
        return {"error": "job_id must be an integer"}
    out = _request("GET", f"/generate/jobs/{jid}")
    if out.get("error"):
        return out
    paths = out.get("result_paths")
    if isinstance(paths, str):
        try:
            paths = json.loads(paths)
        except json.JSONDecodeError:
            paths = [paths]
    return {"job_id": out.get("id", jid),
            "workflow": out.get("workflow_name"),
            "status": out.get("status"),
            "error": out.get("error_message") or None,
            "result_paths": paths,
            "created_at": out.get("created_at"),
            "started_at": out.get("started_at")}


def train_submit(params: dict) -> dict:
    """Submit a LoRA training job (musubi / ACE-Step).

    NON-IDEMPOTENT — confirm with the operator first. Needs name,
    target_arch, dataset_path; see Backlot TrainingCreateRequest for
    the full field set. Returns {job_id}.
    """
    if not isinstance(params, dict) or not params.get("name"):
        return {"error": "params need at least name/target_arch/"
                         "dataset_path"}
    out = _request("POST", "/training/jobs", payload=params, timeout=20)
    if out.get("error"):
        return out
    return {"ok": True, "job_id": out.get("job_id") or out.get("id"),
            "status": out.get("status", "pending")}


def train_jobs(limit: int = 10) -> dict:
    """List recent LoRA training jobs with statuses."""
    out = _request("GET", "/training/jobs", timeout=8)
    rows = out.get("data") if isinstance(out.get("data"), list) else (
        out if isinstance(out, list) else [])
    rows = rows[: max(1, min(int(limit), 25))]
    return {"jobs": [{
        "id": r.get("id"),
        "name": r.get("name"),
        "status": r.get("status"),
        "arch": r.get("target_arch"),
    } for r in rows if isinstance(r, dict)]}


# ---------------------------------------------------------------- studio
# voice/audio lanes on Backlot's live endpoints (Piper/Whisper/demucs).
# speak/transcribe are synchronous and cheap; stems rides the durable
# zaudio_stems worker. Heavy submits stay confirm-gated at the tool layer.


_SPEAK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "speak")


def _b64_data_url(path: str, mime: str) -> str:
    import base64

    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) > 26_214_400:
        raise ValueError(f"file too large ({len(raw)} bytes > 25 MiB)")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def speak(text: str, character: str = "",
          length_scale: float | None = None) -> dict:
    """Piper TTS on Backlot; saves WAV under fabric/data/speak/."""
    text = (text or "").strip()
    if not text:
        return {"error": "missing text"}
    body: dict = {"text": text[:4000]}
    if character:
        body["character"] = character
    if length_scale is not None:
        body["length_scale"] = float(length_scale)
    out = _request("POST", "/generate/speak", payload=body, timeout=60)
    if out.get("error"):
        return out
    audio = out.get("audio") or ""
    if not audio.startswith("data:"):
        return {"error": "backlot returned no audio"}
    try:
        import base64

        payload = audio.split(",", 1)[1]
        raw = base64.b64decode(payload)
    except Exception as e:  # noqa: BLE001
        return {"error": f"bad audio payload: {e}"}
    os.makedirs(_SPEAK_DIR, exist_ok=True)
    fname = f"speak-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    dest = os.path.join(_SPEAK_DIR, fname)
    with open(dest, "wb") as f:
        f.write(raw)
    return {"ok": True, "file": dest, "bytes": len(raw),
            "text": text[:120]}


def transcribe(path: str, model: str | None = None,
               language: str | None = None,
               task: str = "transcribe") -> dict:
    """Whisper STT on Backlot for a local audio/video file."""
    path = str(path or "").strip()
    if not path:
        return {"error": "missing path"}
    if not os.path.isfile(path):
        return {"error": f"not a file: {path}"}
    mime = "audio/wav" if path.lower().endswith(".wav") else \
        "audio/mpeg" if path.lower().endswith((".mp3", ".mp4")) else \
        "application/octet-stream"
    try:
        data_url = _b64_data_url(path, mime)
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    body: dict = {"audio": data_url, "task": task}
    if model:
        body["model"] = model
    if language:
        body["language"] = language
    out = _request("POST", "/generate/transcribe", payload=body,
                   timeout=180)
    if out.get("error"):
        return out
    return {"ok": True, "text": out.get("text", ""),
            "language": out.get("language"),
            "n_segments": len(out.get("segments") or [])}


def stems_models() -> dict:
    """Available demucs models on Backlot."""
    out = _request("GET", "/generate/stems/models")
    rows = out.get("models") if isinstance(out.get("models"), list) else (
        out.get("data") if isinstance(out.get("data"), list) else [])
    models = []
    for r in rows[:20]:
        if isinstance(r, dict):
            models.append({
                "name": r.get("name"),
                "stems": r.get("stems"),
                "default": bool(r.get("default")),
            })
        elif isinstance(r, str):
            models.append({"name": r})
    return {"models": models}


def stems_submit(path: str, model: str = "htdemucs") -> dict:
    """Queue a demucs stem-separation job (durable worker).

    NON-IDEMPOTENT — confirm-gated. Returns {job_id} immediately;
    poll with backlot_job_status.
    """
    path = str(path or "").strip()
    if not os.path.isfile(path):
        return {"error": f"not a file: {path}"}
    try:
        data_url = _b64_data_url(path, "audio/wav"
                                 if path.lower().endswith(".wav")
                                 else "audio/mpeg")
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    out = _request("POST", "/generate/stems",
                   payload={"source_audio": data_url, "model": model},
                   timeout=30)
    if out.get("error"):
        return out
    return {"ok": True, "program": "stems",
            "job_id": out.get("job_id"), "status": "pending",
            "note": f"poll with backlot_job_status({out.get('job_id')})"}


def sample_export(fmt: str, paths: list, name: str = "") -> dict:
    """Export a WAV set / Korg Volca Sample 2 / Ableton Simpler bundle.

    NON-IDEMPOTENT — confirm-gated. fmt: wavset|volca|ableton."""
    fmt = str(fmt or "").strip().lower()
    if fmt not in ("wavset", "volca", "ableton"):
        return {"error": "format must be wavset|volca|ableton"}
    if not isinstance(paths, list) or not paths:
        return {"error": "paths must be a non-empty list"}
    body: dict = {"format": fmt, "paths": [str(p) for p in paths[:64]]}
    if name:
        body["name"] = str(name)[:80]
    out = _request("POST", "/generate/sample/export", payload=body,
                   timeout=120)
    if out.get("error"):
        return out
    return {"ok": True, "file_path": out.get("file_path"),
            "download_url": out.get("download_url")}


def mesh_upgrade(job_id: int, params: dict | None = None) -> dict:
    """Queue a higher-quality re-mesh of a previous mesh job.

    NON-IDEMPOTENT — confirm-gated."""
    try:
        jid = int(job_id)
    except (TypeError, ValueError):
        return {"error": "job_id must be an integer"}
    out = _request("POST", f"/generate/mesh/{jid}/upgrade",
                   payload=params or {}, timeout=30)
    if out.get("error"):
        return out
    return {"ok": True, "program": "mesh-upgrade",
            "job_id": out.get("job_id"), "status": "pending"}


def stepseq(params: dict) -> dict:
    """Queue a ComfyUI step-sequencer image grid.

    NON-IDEMPOTENT — confirm-gated. params: prompt, steps[], width,
    height, copies, seed… (Backlot StepSeqRequest)."""
    if not isinstance(params, dict) or not params.get("prompt"):
        return {"error": "params need prompt and steps[]"}
    out = _request("POST", "/generate/stepseq", payload=params,
                   timeout=30)
    if out.get("error"):
        return out
    return {"ok": True, "program": "stepseq",
            "job_id": out.get("job_id"), "status": "pending"}


def transcode_submit(path: str, quality: int | None = None) -> dict:
    """Transcode a Backlot outputs video to AV1 (Intel QSV).

    SYNCHRONOUS on the Backlot side (ffmpeg run, may take minutes).
    NON-IDEMPOTENT — confirm-gated. path is absolute or relative to
    Backlot's OUTPUTS_DIR."""
    path = str(path or "").strip()
    if not path:
        return {"error": "missing path"}
    body: dict = {"path": path}
    if quality is not None:
        q = int(quality)
        if not 1 <= q <= 63:
            return {"error": "quality must be 1..63"}
        body["quality"] = q
    out = _request("POST", "/generate/video/transcode", payload=body,
                   timeout=900)
    if out.get("error"):
        return out
    return {"ok": True, "output_path": out.get("output_path"),
            "codec": out.get("codec"),
            "size_bytes": out.get("size_bytes"),
            "fallback": bool(out.get("fallback"))}


def transcode_status() -> dict:
    """AV1/QSV transcoder capability flags on Backlot."""
    out = _request("GET", "/generate/video/av1-status")
    return out if out else {}
