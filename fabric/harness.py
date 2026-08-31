"""Harness provider settings — external orchestrator, local workers.

The operator picks one cloud LLM as the *orchestrator* (plan / refine
MiniApps). Duty vLLM and fabric TOOLS stay the workers: tools never run
on the external provider. Keys live in fabric/data/harness.json (gitignored,
mode 0600). GET never returns a raw key.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "harness.json")
_LOCK = threading.Lock()
_CACHE: dict | None = None
_MODELS_TTL = 60.0
_MODELS_CACHE: dict[str, tuple[float, dict]] = {}
_SKIP_MODEL = ("embed", "whisper", "tts", "moderation", "dall-e", "image",
               "audio", "realtime")

# Orchestrator ids the settings panel lists. `local` is duty vLLM.
PROVIDERS = {
    "local": {
        "id": "local",
        "label": "duty vLLM",
        "blurb": "on-box Qwen 3.8 — no external key",
        "base": os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "model": os.environ.get("FABRIC_LLM_MODEL") or "",
        "env": "FABRIC_LLM_API_KEY",
        "signup": "",
        "external": False,
    },
    "grok": {
        "id": "grok",
        "label": "Grok (xAI)",
        "blurb": "SpaceXAI / api.x.ai — orchestrator",
        "base": "https://api.x.ai/v1",
        "model": "grok-4.6",
        "models": ["grok-4.6", "grok-4.5", "grok-4"],
        "env": "XAI_API_KEY",
        "signup": "https://console.x.ai",
        "external": True,
    },
    "kimi": {
        "id": "kimi",
        "label": "Kimi (Moonshot)",
        "blurb": "api.moonshot.ai — orchestrator",
        "base": "https://api.moonshot.ai/v1",
        "model": "kimi-k3",
        "models": ["kimi-k3"],
        "env": "KIMI_API_KEY",
        "env_alt": "MOONSHOT_API_KEY",
        "signup": "https://platform.moonshot.ai",
        "external": True,
    },
    "opencode-zen": {
        "id": "opencode-zen",
        "label": "OpenCode Zen",
        "blurb": "opencode.ai/zen — orchestrator",
        "base": "https://opencode.ai/zen/v1",
        "model": "laguna-s-2.1-free",
        "models": ["laguna-s-2.1-free", "nemotron-3.5-lightning-free",
                   "x-preview-f-free"],
        "env": "OPENCODE_API_KEY",
        "signup": "https://opencode.ai/auth",
        "external": True,
    },
}


def _path() -> str:
    return os.environ.get("FABRIC_HARNESS_PATH", _DEFAULT_PATH)


def _blank() -> dict:
    return {
        "orchestrator": "local",
        "providers": {
            pid: {"api_key": "", "model": meta["model"]}
            for pid, meta in PROVIDERS.items() if pid != "local"
        },
    }


def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = _path()
    data = _blank()
    try:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            orch = raw.get("orchestrator") or "local"
            if orch not in PROVIDERS:
                orch = "local"
            data["orchestrator"] = orch
            saved = raw.get("providers") or {}
            for pid in data["providers"]:
                row = saved.get(pid) or {}
                if isinstance(row, dict):
                    if row.get("api_key"):
                        data["providers"][pid]["api_key"] = str(row["api_key"])
                    if row.get("model"):
                        data["providers"][pid]["model"] = str(row["model"])
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError):
        pass
    _CACHE = data
    return data


def _save(data: dict) -> None:
    global _CACHE
    path = _path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    _CACHE = data


def reset_cache() -> None:
    global _CACHE
    _CACHE = None
    _MODELS_CACHE.clear()


def _mask(key: str) -> dict:
    key = (key or "").strip()
    if not key:
        return {"key_set": False, "key_tail": ""}
    tail = key[-4:] if len(key) >= 4 else key
    return {"key_set": True, "key_tail": tail}


def _env_key(meta: dict) -> str:
    for name in (meta.get("env"), meta.get("env_alt")):
        if name and os.environ.get(name, "").strip():
            return os.environ[name].strip()
    return ""


def _provider_key(pid: str, data: dict | None = None) -> str:
    meta = PROVIDERS[pid]
    data = data or _load()
    if pid == "local":
        return os.environ.get(meta["env"], "local")
    stored = ((data.get("providers") or {}).get(pid) or {}).get("api_key") or ""
    return str(stored).strip() or _env_key(meta)


def _provider_model(pid: str, data: dict | None = None) -> str:
    meta = PROVIDERS[pid]
    data = data or _load()
    if pid == "local":
        return meta["model"] or "duty"
    stored = ((data.get("providers") or {}).get(pid) or {}).get("model") or ""
    return str(stored).strip() or meta["model"]


def _catalog_models(pid: str) -> list[str]:
    meta = PROVIDERS.get(pid) or {}
    ids = list(meta.get("models") or [])
    default = meta.get("model")
    if default and default not in ids:
        ids.insert(0, default)
    return ids


def _parse_model_ids(payload) -> list[str]:
    rows = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            rows = payload["data"]
        elif isinstance(payload.get("models"), list):
            rows = payload["models"]
    elif isinstance(payload, list):
        rows = payload
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            mid = row.strip()
        elif isinstance(row, dict):
            mid = str(row.get("id") or row.get("name") or row.get("model") or "").strip()
        else:
            continue
        if not mid or mid in seen:
            continue
        low = mid.lower()
        if any(tok in low for tok in _SKIP_MODEL):
            continue
        seen.add(mid)
        ids.append(mid)
    return ids


def list_models(pid: str, timeout: float = 4.0) -> dict:
    """GET {base}/models from the provider. Falls back to the catalog seed."""
    if pid not in PROVIDERS:
        return {"ok": False, "provider": pid, "live": False,
                "models": [], "error": "unknown provider"}
    now = time.monotonic()
    hit = _MODELS_CACHE.get(pid)
    if hit and now - hit[0] < _MODELS_TTL:
        return dict(hit[1])
    meta = PROVIDERS[pid]
    catalog = _catalog_models(pid)
    key = _provider_key(pid)
    headers = {"User-Agent": "woodfire-fabric"}
    if key and key != "local":
        headers["Authorization"] = f"Bearer {key}"
    url = meta["base"].rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        ids = _parse_model_ids(payload)
        if not ids:
            out = {"ok": False, "provider": pid, "live": False,
                   "models": catalog, "error": "empty /models", "source": "catalog"}
        else:
            current = _provider_model(pid)
            if current and current not in ids:
                ids = [current] + ids
            out = {"ok": True, "provider": pid, "live": True,
                   "models": ids, "error": None, "source": "live",
                   "count": len(ids)}
    except urllib.error.HTTPError as e:
        out = {"ok": False, "provider": pid, "live": False, "models": catalog,
               "error": f"HTTP {e.code}", "source": "catalog"}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "provider": pid, "live": False, "models": catalog,
               "error": str(e), "source": "catalog"}
    _MODELS_CACHE[pid] = (now, out)
    return dict(out)


def fetch_all_models() -> dict:
    """Parallel /models for every provider. Used by the settings dropdown."""
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(list_models, pid): pid for pid in PROVIDERS}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                rows[pid] = fut.result()
            except Exception as e:  # noqa: BLE001
                rows[pid] = {"ok": False, "provider": pid, "live": False,
                             "models": _catalog_models(pid), "error": str(e),
                             "source": "catalog"}
    return {"providers": rows}


def _vllm_up() -> bool:
    url = os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    url = url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "woodfire-fabric"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status < 400
    except Exception:
        return False


def public() -> dict:
    data = _load()
    orch = data.get("orchestrator") or "local"
    rows = []
    for pid, meta in PROVIDERS.items():
        key = _provider_key(pid, data)
        masked = _mask(key) if meta["external"] else {"key_set": True, "key_tail": ""}
        rows.append({
            "id": pid,
            "label": meta["label"],
            "blurb": meta["blurb"],
            "base": meta["base"],
            "model": _provider_model(pid, data),
            "models": list(meta.get("models") or ([meta["model"]] if meta.get("model") else [])),
            "signup": meta["signup"],
            "external": meta["external"],
            "env": meta["env"],
            **masked,
        })
    return {
        "orchestrator": orch,
        "providers": rows,
        "workers": [
            {
                "id": "vllm",
                "role": "agent worker",
                "url": os.environ.get("FABRIC_LLM_BASE_URL",
                                     "http://127.0.0.1:8000/v1"),
                "up": _vllm_up(),
            },
            {
                "id": "tools",
                "role": "tool runner (fabric MCP/tools)",
                "url": "local",
                "up": True,
            },
        ],
        "hint": ("External LLM orchestrates. Duty vLLM and fabric tools "
                 "execute. Keys stay on the ship; GET never returns them."),
    }


def orchestrator() -> dict:
    """Resolved endpoint for the duty chat loop."""
    data = _load()
    pid = data.get("orchestrator") or "local"
    if pid not in PROVIDERS:
        pid = "local"
    meta = PROVIDERS[pid]
    model = _provider_model(pid, data)
    if pid == "local":
        model = os.environ.get("FABRIC_LLM_MODEL") or None
        base = os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        key = os.environ.get("FABRIC_LLM_API_KEY", "local")
    else:
        base = meta["base"]
        key = _provider_key(pid, data)
    return {
        "id": pid,
        "label": meta["label"],
        "base": base.rstrip("/"),
        "model": model,
        "api_key": key,
        "external": meta["external"],
        "ready": bool(key) if meta["external"] else True,
    }


def update(body: dict) -> dict:
    if not isinstance(body, dict):
        return {"error": "empty body"}
    with _LOCK:
        data = _load()
        providers = body.get("providers")
        if isinstance(providers, dict):
            for pid, row in providers.items():
                if pid not in data["providers"] or not isinstance(row, dict):
                    continue
                if "api_key" in row:
                    key = row["api_key"]
                    if key is None:
                        data["providers"][pid]["api_key"] = ""
                    elif str(key).strip():
                        data["providers"][pid]["api_key"] = str(key).strip()
                if row.get("model"):
                    data["providers"][pid]["model"] = str(row["model"]).strip()
        else:
            for pid in list(data["providers"]):
                raw = body.get(pid + "_key")
                if raw is None:
                    continue
                if str(raw).strip() == "":
                    continue
                if str(raw).strip() == "-":
                    data["providers"][pid]["api_key"] = ""
                else:
                    data["providers"][pid]["api_key"] = str(raw).strip()
            for pid in list(data["providers"]):
                model = body.get(pid + "_model")
                if model:
                    data["providers"][pid]["model"] = str(model).strip()
        orch = body.get("orchestrator")
        if orch:
            if orch not in PROVIDERS:
                return {"error": f"unknown orchestrator {orch}"}
            if PROVIDERS[orch]["external"] and not _provider_key(orch, data):
                return {"error": f"{orch} has no API key yet — paste it first"}
            data["orchestrator"] = orch
        _save(data)
        _MODELS_CACHE.clear()
    return public()


def api_kind(pid: str, model: str) -> str:
    """Which HTTP shape OpenCode Zen (and lookalikes) want for this id."""
    if pid != "opencode-zen":
        return "chat"
    m = (model or "").lower()
    if m.startswith("gemini-"):
        return "unsupported"
    if m.startswith(("gpt-", "grok-", "muse-spark")):
        return "responses"
    if m.startswith("claude-") or m.startswith("qwen"):
        return "messages"
    return "chat"


def _headers(key: str, kind: str) -> dict:
    h = {"Content-Type": "application/json",
         "Authorization": f"Bearer {key}",
         "User-Agent": "woodfire-fabric"}
    if kind == "messages":
        h["anthropic-version"] = "2023-06-01"
        h["x-api-key"] = key
    return h


def _text_chat(data: dict) -> str:
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def _text_responses(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    if data.get("output_text"):
        return str(data["output_text"]).strip()
    parts = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "output_text" and item.get("text"):
            parts.append(item["text"])
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"])
    return "".join(parts).strip()


def _text_messages(data: dict) -> str:
    content = (data or {}).get("content")
    if isinstance(content, list):
        return "".join(
            (c.get("text") or "") for c in content if isinstance(c, dict)).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    sys_parts = []
    rest = []
    for m in messages or []:
        if m.get("role") == "system":
            sys_parts.append(m.get("content") or "")
        else:
            rest.append({"role": m.get("role") or "user",
                         "content": m.get("content") or ""})
    if not rest:
        rest = [{"role": "user", "content": "ping"}]
    return "\n".join(sys_parts).strip(), rest


_DEFAULT_MAX_TOKENS = 4096
_MAX_LEGS = 3
_TOKENS_CEILING = 32768


def _default_max_tokens() -> int:
    raw = os.environ.get("FABRIC_LLM_MAX_TOKENS", "")
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_MAX_TOKENS
    return max(256, min(n, _TOKENS_CEILING))


def _join_fragments(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    glue = "" if (a[-1].isspace() or b[0].isspace()
                  or b[0] in ",.;:!?)]:}") else " "
    return a + glue + b


def _length_truncated(kind: str, data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if kind == "chat":
        choices = data.get("choices") or [{}]
        first = choices[0] if choices else {}
        return isinstance(first, dict) and first.get("finish_reason") == "length"
    if kind == "messages":
        return data.get("stop_reason") == "max_tokens"
    if data.get("status") == "incomplete":
        det = data.get("incomplete_details")
        reason = det.get("reason") if isinstance(det, dict) else None
        return reason != "content_filter"
    return False


_CONTINUE_NOTE = ("Continue exactly where you stopped. Do not repeat any "
                  "earlier text.")


def post_llm(base: str, api_key: str, model: str, messages: list[dict],
             orch: str = "local", timeout: float = 90.0,
             max_tokens: int | None = None) -> str:
    """One completion against chat, Zen responses, or Anthropic messages.

    Length-capped replies are auto-continued (up to two follow-up legs) so
    the caller never receives a silent mid-sentence cut.
    """
    kind = api_kind(orch, model)
    if kind == "unsupported":
        raise RuntimeError(
            f"{orch} {model} is not a chat endpoint (Gemini on Zen). "
            "Pick a chat/completions, GPT/Grok, or Claude id.")
    limit = int(max_tokens or _default_max_tokens())
    base = (base or "").rstrip("/")
    system, rest = _split_system(messages)
    text = ""
    for _leg in range(_MAX_LEGS):
        if kind == "responses":
            payload = {"model": model, "input": rest,
                       "max_output_tokens": limit}
            if system:
                payload["instructions"] = system
            path = "/responses"
            extract = _text_responses
        elif kind == "messages":
            payload = {"model": model, "max_tokens": limit, "messages": rest,
                       "temperature": 0.2}
            if system:
                payload["system"] = system
            path = "/messages"
            extract = _text_messages
        else:
            convo = ([{"role": "system", "content": system}]
                     if system else []) + rest
            # Model-card thinking-mode sampling (Qwen3.8-27B): the old
            # temperature=0.2 fought the card (temp 1.0, top_p 0.95,
            # top_k 20) and sharpened the repetition failure modes.
            # Send the card set explicitly rather than relying on the
            # serve's generation_config defaults.
            payload = {"model": model, "messages": convo,
                       "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                       "max_tokens": limit}
            path = "/chat/completions"
            extract = _text_chat
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers=_headers(api_key, kind),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode()[:800]
            except Exception:
                raw = ""
            msg = raw or (e.reason or "error")
            try:
                parsed = json.loads(raw)
                err = parsed.get("error") if isinstance(parsed, dict) else None
                if isinstance(err, dict):
                    msg = err.get("message") or err.get("type") or msg
                elif isinstance(err, str):
                    msg = err
            except Exception:
                pass
            raise RuntimeError(f"{orch} {model} HTTP {e.code}: {msg}") from e
        piece = extract(data) if isinstance(data, dict) else ""
        text = _join_fragments(text, piece)
        if not piece or not _length_truncated(kind, data):
            break
        rest = rest + [{"role": "assistant", "content": piece},
                       {"role": "user", "content": _CONTINUE_NOTE}]
    return text


def ping(pid: str | None = None) -> dict:
    """One cheap request against the chosen (or named) orchestrator."""
    data = _load()
    target = pid or data.get("orchestrator") or "local"
    if target not in PROVIDERS:
        return {"ok": False, "error": f"unknown provider {target}"}
    if target == "local":
        up = _vllm_up()
        return {"ok": up, "provider": target,
                "error": None if up else "duty vLLM not reachable"}
    key = _provider_key(target, data)
    if not key:
        return {"ok": False, "provider": target, "error": "no API key"}
    meta = PROVIDERS[target]
    model = _provider_model(target, data)
    try:
        text = post_llm(meta["base"], key, model,
                        [{"role": "user", "content": "ping"}],
                        orch=target, timeout=20)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "provider": target, "model": model,
                "error": str(e)}
    if not text:
        return {"ok": False, "provider": target, "model": model,
                "error": "empty reply (endpoint up, model returned nothing)"}
    return {"ok": True, "provider": target, "model": model, "reply": text[:200]}
