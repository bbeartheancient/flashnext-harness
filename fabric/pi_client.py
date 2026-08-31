"""Pi coding-agent client — olympus-pi-bridge on :8642.

Pi is the coding/research lane (harness-plan). Duty vLLM stays on :8000
for chat/tools. This module does not spawn Pi; it POSTs to the live bridge.

Duty vLLM 400s OpenAI tool_choice=auto, so Pi dumps XML instead of running
native tools. Fabric executes a short sandboxed loop (ls/grep/find/read
under the working tree, plus generate_microapp) and re-prompts.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

_DEFAULT = "http://127.0.0.1:8642"
_MAX_ROUNDS = 4
_MAX_CALLS = 4
_LS_CAP = 200
_GREP_HITS = 40
_FIND_CAP = 80
_READ_BYTES = 80_000
_READ_LINES = 200
_RESULT_CAP = 6000

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", ".venv-fabric",
    ".venv-vllm", "build", "build-comfy", "build-onednn", "cache",
    "third_party", "vendor", "logs", ".grok",
}
_SKIP_SUFFIX = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".woff", ".woff2",
    ".so", ".a", ".o", ".bin", ".pt", ".safetensors", ".gguf",
    ".zip", ".tar", ".gz", ".mp4", ".pyc", ".whl",
}

_TOOL_RE = re.compile(
    r'<tool\s+name=["\']([A-Za-z0-9_]+)["\']\s*>(.*?)</tool>',
    re.S,
)
_FN_CLOSE_RE = re.compile(
    r'<function=([A-Za-z0-9_]+)>(.*?)</function>',
    re.S,
)
_PARAM_RE = re.compile(
    r'<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>',
    re.S,
)
_BARE_FN_RE = re.compile(
    r'<function=([A-Za-z0-9_]+)(?:\s+([^<>]*))?\s*/?>',
)
_HTML_FENCE = re.compile(r"```html\s*(.*?)```", re.S | re.I)
_MINIAPP_HINT = re.compile(
    r"mini-?app|simulator|visuali[sz]er|tracker|widget|game\b", re.I)

_ALLOWED = {"ls", "grep", "find", "read", "write", "edit",
            "generate_microapp", "list_microapps"}
_REFUSED = {"bash", "rm", "delete"}
_DENY_FILES = {"harness.json", ".env"}
_WRITE_MAX = 200_000
_INSPECT = {"ls", "grep", "find", "read"}


def _base() -> str:
    return os.environ.get("FABRIC_PI_URL", _DEFAULT).rstrip("/")


def _repo() -> Path:
    return Path(os.environ.get("FABRIC_PI_CWD", str(Path.home()))).resolve()


def _duty_model() -> str:
    env = os.environ.get("FABRIC_PI_MODEL")
    if env:
        return env
    try:
        req = urllib.request.Request(
            os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
            + "/models",
            headers={"User-Agent": "woodfire-fabric"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data") or []
        if models:
            return str(models[0]["id"])
    except Exception:
        pass
    return "default"


def health() -> dict:
    url = _base() + "/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "woodfire-fabric"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
        data["up"] = True
        data["url"] = _base()
        return data
    except Exception as e:  # noqa: BLE001
        return {"up": False, "url": _base(), "error": str(e),
                "bridge": "olympus-pi-bridge", "agent": "pi"}


def _strip_think(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _jsonish(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val
    except json.JSONDecodeError:
        return raw


def _args_from_body(body: str) -> dict:
    args: dict = {}
    keys = re.findall(r"<arg_key>(.*?)</arg_key>", body or "", re.S)
    vals = re.findall(r"<arg_value>(.*?)</arg_value>", body or "", re.S)
    for k, v in zip(keys, vals):
        args[k.strip()] = _jsonish(v)
    for m in _PARAM_RE.finditer(body or ""):
        args[m.group(1)] = _jsonish(m.group(2))
    inner = (body or "").strip()
    if not args and inner.startswith("{"):
        parsed = _jsonish(inner)
        if isinstance(parsed, dict):
            return parsed
    if not args and inner and "<" not in inner[:1]:
        args["_raw"] = inner[:500]
    return args


def _bare_rest_args(name: str, rest: str) -> dict:
    rest = (rest or "").strip()
    if not rest:
        return {}
    if name == "grep":
        return {"pattern": rest}
    if name == "find":
        return {"name": rest}
    return {"path": rest}


def parse_pi_tools(text: str) -> list[tuple[str, dict]]:
    """All duty-XML, Qwen <function=>, and bare <function=ls> calls."""
    if not text:
        return []
    found: list[tuple[int, str, dict]] = []
    covered: list[tuple[int, int]] = []

    def overlap(span: tuple[int, int]) -> bool:
        a, b = span
        return any(a >= s and b <= e for s, e in covered)

    for m in _TOOL_RE.finditer(text):
        body = (m.group(2) or "").strip() or "{}"
        parsed = _jsonish(body)
        args = parsed if isinstance(parsed, dict) else _args_from_body(body)
        found.append((m.start(), m.group(1), args))
        covered.append(m.span())
    for m in _FN_CLOSE_RE.finditer(text):
        found.append((m.start(), m.group(1), _args_from_body(m.group(2) or "")))
        covered.append(m.span())
    for m in _BARE_FN_RE.finditer(text):
        if overlap(m.span()):
            continue
        name = m.group(1)
        rest = (m.group(2) or "").strip()
        after = text[m.end(): m.end() + 800]
        stop = re.search(r"</function>|<function=|<tool\s", after)
        chunk = after[: stop.start()] if stop else after
        args = _args_from_body(chunk)
        extra = _bare_rest_args(name, rest)
        extra.update(args)
        found.append((m.start(), name, extra))
    found.sort(key=lambda row: row[0])
    return [(name, args) for _, name, args in found]


def _safe_path(p: str | None) -> Path:
    repo = _repo()
    raw = (p or ".").strip() or "."
    cand = Path(raw)
    if not cand.is_absolute():
        cand = repo / cand
    cand = cand.resolve()
    repo_s = str(repo)
    cand_s = str(cand)
    if cand_s != repo_s and not cand_s.startswith(repo_s + os.sep):
        raise ValueError(f"path outside working tree: {raw}")
    return cand


def _skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".venv")


def _trim(obj, limit: int = _RESULT_CAP) -> str:
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, default=str)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _tool_ls(args: dict) -> str:
    path = _safe_path(str(args.get("path") or args.get("_raw") or "."))
    if path.is_file():
        return str(path.relative_to(_repo()))
    if not path.is_dir():
        return f"error: not found {path}"
    names = []
    for entry in sorted(path.iterdir(), key=lambda e: e.name.lower()):
        if _skip_dir(entry.name):
            continue
        names.append(entry.name + ("/" if entry.is_dir() else ""))
        if len(names) >= _LS_CAP:
            names.append("…")
            break
    return "\n".join(names) or "(empty)"


def _walk_files(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        for name in filenames:
            yield Path(dirpath) / name


def _tool_grep(args: dict) -> str:
    pattern = args.get("pattern") or args.get("query") or args.get("regex") or args.get("_raw") or ""
    pattern = str(pattern).strip()
    if not pattern:
        return "error: missing pattern"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: bad pattern ({e})"
    path = _safe_path(str(args.get("path") or "."))
    hits: list[str] = []
    repo = _repo()
    for fp in _walk_files(path):
        if fp.suffix.lower() in _SKIP_SUFFIX:
            continue
        try:
            if fp.stat().st_size > 1_000_000:
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = fp.relative_to(repo)
                hits.append(f"{rel}:{i}:{line.strip()[:200]}")
                if len(hits) >= _GREP_HITS:
                    hits.append("…")
                    return "\n".join(hits)
    return "\n".join(hits) or "(no matches)"


def _tool_find(args: dict) -> str:
    name = str(args.get("name") or args.get("pattern") or args.get("_raw") or "*")
    path = _safe_path(str(args.get("path") or "."))
    repo = _repo()
    hits: list[str] = []
    for fp in _walk_files(path):
        rel = str(fp.relative_to(repo))
        if fnmatch(fp.name, name) or fnmatch(rel, name) or fp.name == name:
            hits.append(rel)
            if len(hits) >= _FIND_CAP:
                hits.append("…")
                break
    return "\n".join(hits) or "(none)"


def _tool_read(args: dict) -> str:
    raw = args.get("path") or args.get("file") or args.get("_raw") or ""
    if not raw:
        return "error: missing path"
    path = _safe_path(str(raw))
    if not path.is_file():
        return f"error: not a file {path}"
    if path.suffix.lower() in _SKIP_SUFFIX:
        return f"error: binary suffix {path.suffix}"
    try:
        offset = int(args.get("offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(args.get("limit") or _READ_LINES)
    except (TypeError, ValueError):
        limit = _READ_LINES
    text = path.read_text(encoding="utf-8", errors="replace")[:_READ_BYTES]
    lines = text.splitlines()
    slice_ = lines[max(offset, 0): max(offset, 0) + max(limit, 1)]
    rel = path.relative_to(_repo())
    body = "\n".join(slice_)
    return f"{rel} lines {offset + 1}-{offset + len(slice_)}/{len(lines)}\n{body}"


def _writable(path: Path) -> Path:
    if path.name in _DENY_FILES or path.suffix.lower() in {".pem", ".key"}:
        raise ValueError(f"refused secret file {path.name}")
    if path.suffix.lower() in _SKIP_SUFFIX:
        raise ValueError(f"binary suffix {path.suffix}")
    return path


def _tool_write(args: dict) -> str:
    raw = args.get("path") or args.get("file") or ""
    if not raw:
        return "error: missing path"
    contents = args.get("contents")
    if contents is None:
        contents = args.get("content") or args.get("text") or ""
    contents = str(contents)
    if len(contents) > _WRITE_MAX:
        return f"error: contents too large ({len(contents)} > {_WRITE_MAX})"
    path = _writable(_safe_path(str(raw)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    rel = path.relative_to(_repo())
    return f"wrote {rel} ({len(contents)} bytes, {contents.count(chr(10))+1} lines)"


def _tool_edit(args: dict) -> str:
    raw = args.get("path") or args.get("file") or ""
    old = args.get("old") or args.get("old_string") or args.get("search") or ""
    new = args.get("new") or args.get("new_string") or args.get("replace")
    if not raw or not old:
        return "error: edit needs path and old"
    if new is None:
        return "error: edit needs new"
    path = _writable(_safe_path(str(raw)))
    if not path.is_file():
        return f"error: not a file {path}"
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        return "error: old string not found"
    if n > 1:
        return f"error: old string matches {n} times — make it unique"
    path.write_text(text.replace(old, str(new), 1), encoding="utf-8")
    rel = path.relative_to(_repo())
    return f"edited {rel}"


def _dispatch(name: str, args: dict):
    if name in _REFUSED:
        return ("refused: bash/rm stay off this lane. Use write/edit for files "
                "inside the working tree.")
    if name not in _ALLOWED:
        return f"unknown tool {name}. allowed: {sorted(_ALLOWED)}"
    try:
        if name == "ls":
            return _tool_ls(args)
        if name == "grep":
            return _tool_grep(args)
        if name == "find":
            return _tool_find(args)
        if name == "read":
            return _tool_read(args)
        if name == "write":
            return _tool_write(args)
        if name == "edit":
            return _tool_edit(args)
        if name == "list_microapps":
            from . import microapps
            return {"apps": microapps.list_public()}
        if name == "generate_microapp":
            from . import microapps
            query = str(args.get("query") or args.get("title") or "miniapp")
            html = args.get("html") or None
            if isinstance(html, str):
                html = html.strip()
                if len(html) < 40 or not html.strip(" .…"):
                    html = None
            if not html:
                kn = microapps.pick_kernel(query)
                if kn not in microapps._KERNELS:
                    return ("error: no kernel for this MiniApp; resend "
                            "generate_microapp with html= compact <!DOCTYPE html> "
                            "canvas + JS, addEventListener('message'), no remote scripts")
            route = None
            if args.get("dept") or args.get("input") or args.get("output"):
                route = {
                    "dept": args.get("dept") or "command",
                    "inputs": [args.get("input") or "text"],
                    "outputs": [args.get("output") or "html"],
                    "viewports": int(args.get("viewports") or 1),
                }
            return microapps.compose(
                query, html=html, title=args.get("title") or None, route=route,
            )
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    return f"unhandled {name}"


def _empty_figure() -> dict:
    return {"type": "html", "html": "<pre class='miniapp-text'></pre>",
            "caption": "pi", "sandbox": False}


def _http_complete(prompt: str, timeout: float = 90.0) -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    h = health()
    if not h.get("up"):
        return {"ok": False, "error": "pi-bridge down", "health": h}
    body = json.dumps({
        "model": _duty_model(),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        _base() + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "woodfire-fabric"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"pi HTTP {e.code}",
                "detail": e.read().decode()[:1500]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    try:
        text = _strip_think(data["choices"][0]["message"]["content"] or "")
    except Exception:
        text = json.dumps(data)[:2000]
    if not text:
        return {
            "ok": False,
            "error": "pi empty reply (bridge used a model id vLLM does not serve)",
            "model": data.get("model"),
            "id": data.get("id"),
            "lane": "pi",
            "figure": _empty_figure(),
        }
    return {
        "ok": True,
        "lane": "pi",
        "reply": text,
        "model": data.get("model"),
        "id": data.get("id"),
        "figure": _empty_figure(),
    }


def _pack(base: str, history: list[tuple[str, str]]) -> str:
    parts = [base]
    for asst, tools in history:
        parts.append("Assistant previously said:\n" + asst[:2500])
        parts.append("Tool results:\n" + tools)
    if history:
        parts.append(
            "Continue: emit another <tool name=…>{json}</tool> block, or the "
            "final answer as plain text. Do not repeat the same call."
        )
    return "\n\n".join(parts)


def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    letters = re.findall(r"[a-zA-Z0-9]", t)
    if not letters:
        return True
    uniq = {c.lower() for c in letters}
    if len(t) < 8 and len(uniq) <= 3:
        return True
    if len(uniq) <= 2 and len(t) < 24:
        return True
    return False


def _summarize_loop(reply: str, used: list[str], shipped: bool,
                    miniapp: bool) -> str:
    text = (reply or "").strip()
    if not _looks_garbage(text):
        if miniapp and not shipped and "generate_microapp" in used:
            return text + ("\n\nPi called generate_microapp without HTML "
                           "(512-token cap). Use /loop on duty, or paste HTML.")
        return text
    bits = []
    if used:
        bits.append("tools: " + ", ".join(used))
    if shipped:
        bits.append("MiniApp registered")
    elif miniapp:
        bits.append("no MiniApp HTML (local 512-token cap); try /loop on duty")
    return "; ".join(bits) or text or "(empty pi reply)"


def _harvest_html(text: str, query: str) -> dict | None:
    m = _HTML_FENCE.search(text or "")
    if not m:
        return None
    html = m.group(1).strip()
    if len(html) < 40:
        return None
    from . import microapps
    return microapps.compose(query, html=html)


def complete(prompt: str, timeout: float = 90.0, tool_loop: bool = True,
             miniapp: bool = False) -> dict:
    """Non-streaming /v1/chat/completions against the Pi bridge."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    if not tool_loop:
        return _http_complete(prompt, timeout=timeout)

    history: list[tuple[str, str]] = []
    used: list[str] = []
    last: dict = {}
    last_figure = None
    last_microapp = None
    for round_i in range(_MAX_ROUNDS):
        last = _http_complete(_pack(prompt, history), timeout=timeout)
        if not last.get("ok"):
            last["tools"] = used
            last["rounds"] = round_i
            return last
        text = last["reply"]
        calls = parse_pi_tools(text)
        if not calls:
            want_app = (
                last_microapp is None
                and miniapp
                and round_i + 1 < _MAX_ROUNDS
            )
            if want_app:
                history.append((
                    text[:800],
                    "Inspection complete. Do not ls/read/find again. Emit now:\n"
                    '<tool name="generate_microapp">{"query":"…","html":"<!DOCTYPE html>…"}</tool>\n'
                    "Compact canvas + JS, no remote scripts, under 350 tokens.",
                ))
                continue
            harvested = None
            if last_microapp is None:
                harvested = _harvest_html(text, prompt)
            if harvested and not harvested.get("error"):
                last_microapp = harvested
                if harvested.get("figure"):
                    last_figure = harvested["figure"]
                used.append("generate_microapp")
            break
        inspect_n = sum(1 for n in used if n in _INSPECT)
        if (
            inspect_n >= 3
            and last_microapp is None
            and miniapp
            and all(n in _INSPECT for n, _ in calls)
        ):
            calls = []
            history.append((
                text[:800],
                "Enough inspection. Emit generate_microapp with compact HTML now.",
            ))
            continue
        chunks = []
        for name, args in calls[:_MAX_CALLS]:
            used.append(name)
            result = _dispatch(name, args)
            if (name == "generate_microapp" and isinstance(result, dict)
                    and not result.get("error") and not result.get("clarify")):
                if result.get("figure"):
                    last_figure = result["figure"]
                last_microapp = result
            chunks.append(f"{name}({json.dumps(args, default=str)[:400]}):\n{_trim(result)}")
        history.append((text, "\n\n".join(chunks)))
    else:
        last["reply"] = (last.get("reply") or "") + "\n\n[tool loop capped at %d]" % _MAX_ROUNDS

    last["tools"] = used
    last["rounds"] = min(len(history) + 1, _MAX_ROUNDS)
    if last_figure:
        last["figure"] = last_figure
    if last_microapp:
        last["microapp"] = last_microapp.get("spec") or last_microapp
        last["id_app"] = last_microapp.get("id")
    if last.get("ok"):
        last["reply"] = _summarize_loop(
            last.get("reply") or "", used, last_microapp is not None, miniapp)
    return last


def code_task(prompt: str, goal: str | None = None) -> dict:
    """Run a coding task via OMP (oh-my-pi).

    The pi XML tool-loop lane was retired 2026-08-27 — OMP is the
    coding agent on the duty 27B.  This keeps the /api/pi/code route
    shape but delegates; the legacy complete() loop below is unused.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    task = prompt if not goal else f"Active goal: {goal}\nTask: {prompt}"
    from .tools import omp_code
    out = omp_code(task)
    if "error" in out and not out.get("ok"):
        return {"ok": False, "error": out["error"], "lane": "omp"}
    return {"ok": out.get("ok", False),
            "reply": out.get("result") or "(no OMP reply)",
            "session": out.get("session"), "lane": "omp"}


def run_goal(statement: str) -> dict:
    statement = (statement or "").strip()
    if not statement:
        return {"error": "empty goal"}
    from .log_store import get_log_store

    store = get_log_store()
    row = store.goal_add(statement)
    out = code_task("Work this goal in the repo. Small goals: do them. "
                    "Large goals: at most an 8-step plan and only the "
                    "first safe step.", goal=statement)
    status = "done" if out.get("ok") else "error"
    store.goal_update(row["id"], status=status,
                      last_result=(out.get("reply") or out.get("error") or "")[:4000])
    out["goal"] = row
    out["goal_status"] = status
    return out
