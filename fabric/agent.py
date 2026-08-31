"""Duty-LLM agent. vLLM on this box does not have --enable-auto-tool-choice
(OpenAI `tool_choice=auto` 400s), so tools are a prompt-side XML loop rather
than the Chat Completions `tools` parameter. pydantic-ai Agent is still used
in unit tests via TestModel override of `_agent`.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request

from .tools import TOOLS

_TOOL_RE = re.compile(
    r'<tool\s+name=["\']([A-Za-z0-9_]+)["\']\s*>(.*?)</tool>',
    re.S,
)
_QWEN_FN_RE = re.compile(
    r'<function=([A-Za-z0-9_]+)>(.*?)</function>',
    re.S,
)


def _base_url() -> str:
    return os.environ.get("FABRIC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def discover_model() -> str:
    env = os.environ.get("FABRIC_LLM_MODEL")
    if env and env != "unknown":
        return env
    try:
        req = urllib.request.Request(
            _base_url() + "/models", headers={"User-Agent": "woodfire-fabric"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data") or []
        if models:
            return str(models[0]["id"])
    except Exception:
        pass
    return "unknown"


def _catalog() -> str:
    lines = []
    for fn in TOOLS:
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"- {fn.__name__}{sig}: {doc}")
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are the duty officer and coding TUI of a local ship computer. "
    "The command bar is your prompt — the operator does not need /chat. "
    "Do the work. Use tools. Edit the tree. Do not stop at a plan unless asked.\n"
    "Working tree: set by FABRIC_PI_CWD (fabric/ is this harness). "
    "Inspect with ls/grep/find/read; change files with write/edit. "
    "Do not touch harness.json, .env, ComfyUI, duty vLLM, or vendor hoa64.\n"
    "Ship calculators (hadamard_build, terrain_los, sage_eval, …) stay tools. "
    "New MiniApps: generate_microapp. Voice-leg: pipecat_status only.\n"
    "CODING TASKS: for multi-step code changes, debugging, or refactoring "
    "call omp_code(task) — it delegates to a dedicated coding agent with "
    "full file/edit/bash tools. Do NOT use for status/lab/media queries.\n"
    "To locate code/docs use code_index(query) first — it maps names to "
    "file:line so you rarely need grep/read sweeps.\n"
    "You orchestrate; tools run locally. Never claim a tool ran unless a "
    "result came back.\n\n"
    "To call a tool, emit exactly one block and nothing else:\n"
    '<tool name="tool_name">{"arg": "value"}</tool>\n'
    "After a tool result arrives, either call another tool the same way or "
    "give the final answer as plain text (no tool tag).\n\n"
    "Available tools:\n"
)


def _strip_think(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _dangling_tool(text: str) -> bool:
    t = text or ""
    opens = len(re.findall(r"<tool\s", t)) + len(re.findall(r"<function=", t))
    closes = t.count("</tool>") + t.count("</function>")
    return opens > closes


_MAX_ROUNDS_DEFAULT = 24
_CTX_CHARS_DEFAULT = 84_000
_TOOL_RESULT_KEEP = 400
_ASSIST_KEEP = 800


def _env_int(name: str, default: int, floor: int) -> int:
    try:
        return max(floor, int(os.environ.get(name, "")))
    except ValueError:
        return default


def _round_cap() -> int:
    return _env_int("FABRIC_AGENT_MAX_ROUNDS", _MAX_ROUNDS_DEFAULT, 4)


def _char_budget() -> int:
    return _env_int("FABRIC_CTX_CHAR_BUDGET", _CTX_CHARS_DEFAULT, 12_000)


def _history_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _fit_history(messages: list[dict], budget: int) -> list[dict]:
    """Shrink a tool-loop conversation to about budget chars.

    Pass 1 elides old tool payloads and assistant turns (conversation
    structure kept); pass 2 keeps system + first user + the newest tail
    that fits. Recent turns stay verbatim.

    Assistant turns carrying a complete tool block keep the BLOCK, not
    the head: head-truncation used to cut the model's reasoning-style
    preamble and drop every intact <tool>/<function= exemplar (or
    worse, leave a dangling opener), and the model stopped emitting
    tool calls once the run crossed the fit budget.
    """
    if _history_chars(messages) <= budget:
        return list(messages)
    out = [dict(m) for m in messages]
    last_i = len(out) - 1
    for i, m in enumerate(out):
        if i >= last_i - 3:
            break
        role = m.get("role")
        c = m.get("content") or ""
        if role == "user" and c.startswith("Tool ") and len(c) > _TOOL_RESULT_KEEP:
            head = c.split("\n", 1)[0]
            tail = ("\nGive the final answer, or call another tool."
                    if c.rstrip().endswith("another tool.") else "")
            m["content"] = head + "\n[earlier tool output elided]" + tail
        elif role == "assistant" and len(c) > _ASSIST_KEEP:
            block = _TOOL_RE.search(c) or _QWEN_FN_RE.search(c)
            if block is not None:
                m["content"] = ("[reasoning elided]\n" + block.group(0)
                                + "\n[earlier assistant turn elided]")
            else:
                m["content"] = c[:200] + "\n[earlier assistant turn elided]"
    if _history_chars(out) <= budget:
        return out
    sys_m = out[0] if out and out[0].get("role") == "system" else None
    rest = out[1:] if sys_m else out
    if len(rest) < 3:
        return out
    keep = [rest[0]]
    for m in reversed(rest[1:-1]):
        cand = ([sys_m] if sys_m else []) + keep + [m] + [rest[-1]]
        if _history_chars(cand) > budget:
            break
        keep.insert(1, m)
    return ([sys_m] if sys_m else []) + keep + [rest[-1]]


def _is_ctx_error(msg: str) -> bool:
    low = (msg or "").lower()
    return ("maximum context length" in low or "context length" in low
            or "too long" in low or "reduce the length" in low)


def _ngram_live_feed(message: str, output: str, emit) -> None:
    """Fire-and-forget per-turn feed into the session's rolling edge
    buffer (FABRIC_NGRAM_LIVE=1).  Runs in a daemon thread — the reply
    path never waits on capture+GEMM (~0.5-2 s warm).

    User message only: storing the assistant's own output created a
    self-referential echo — live edge rows match the tokens the model
    is re-decoding almost exactly, pulling generation back onto its
    own recent text (intra-sentence stutter), amplified by pinning."""
    import threading

    def _work() -> None:
        try:
            from . import latent
            note = latent.live_note(
                latent.live_session(),
                f"User: {message}")
            if note is not None:
                emit({"t": "ngram", "live": True, **note})
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_work, daemon=True).start()


def _ngram_transfer_elided(messages: list[dict]) -> dict | None:
    """FABRIC_NGRAM_COMPACT=1: instead of silently dropping the middle
    of an over-budget conversation, transfer it into the writable
    n-gram PLE (hidden capture → edge GEMM → engine hot-refresh), so
    the post-compaction forward carries the elided context as a
    layer-63 residual delta.  Best-effort; never blocks the fit."""
    if os.environ.get("FABRIC_NGRAM_COMPACT") != "1":
        return None
    try:
        from . import latent
        # The middle is what _fit_history elides/drops: everything but
        # system, the first user turn, and the recent tail.
        rest = (messages[1:] if messages
                and messages[0].get("role") == "system" else messages)
        middle = rest[1:-4] if len(rest) > 5 else []
        if not middle:
            return None
        return latent.ngram_transfer(
            latent._render_transcript(middle), mode="merge")
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a prompt-side tool call. Prefer <tool name=…>{json}</tool>."""
    if not text:
        return None
    m = _TOOL_RE.search(text)
    if m:
        raw = (m.group(2) or "").strip() or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        return m.group(1), args
    m = _QWEN_FN_RE.search(text)
    if m:
        args = {}
        body = m.group(2) or ""
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", body)
        vals = re.findall(r"<arg_value>(.*?)</arg_value>", body)
        for k, v in zip(keys, vals):
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        inner = body.strip()
        if not args and inner.startswith("{"):
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                pass
        return m.group(1), args
    return None


class ToolCallPart:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name


class _Msg:
    def __init__(self, tools: list[str]):
        self.parts = [ToolCallPart(n) for n in tools]


class DutyResult:
    def __init__(self, output: str, tools: list[str], last_result=None):
        self.output = output
        self._tools = tools
        self.last_result = last_result

    def all_messages(self):
        return [_Msg(self._tools)]


def _dispatch(name: str, args: dict):
    table = {fn.__name__: fn for fn in TOOLS}
    fn = table.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _trim(obj, limit=6000) -> str:
    text = json.dumps(obj, default=str)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _provider_http_error(exc: urllib.error.HTTPError, orch: str, model: str) -> str:
    raw = ""
    try:
        raw = exc.read().decode()[:800]
    except Exception:
        raw = ""
    msg = raw or (exc.reason or "error")
    try:
        data = json.loads(raw)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or msg
        elif isinstance(err, str):
            msg = err
        elif isinstance(data, dict) and isinstance(data.get("message"), str):
            msg = data["message"]
    except Exception:
        pass
    return f"{orch} {model} HTTP {exc.code}: {msg}"


class DutyAgent:
    """Prompt-side tool loop against an OpenAI-compatible /v1/chat/completions."""

    def __init__(self, model_name: str | None = None, base: str | None = None,
                 api_key: str | None = None, orchestrator: str = "local"):
        self.orchestrator = orchestrator
        self.model = model_name or discover_model()
        self.base = (base or _base_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("FABRIC_LLM_API_KEY", "local")

    def _complete(self, messages: list[dict]) -> str:
        from . import harness

        # The agent is a module-level singleton; if it was built while
        # the duty serve was down, self.model is "unknown" forever.
        # Re-discover lazily instead of failing every request until a
        # fabric restart.
        if not self.model or self.model == "unknown":
            self.model = discover_model()
        try:
            timeout = float(os.environ.get("FABRIC_LLM_TIMEOUT", "") or 150)
        except ValueError:
            timeout = 150
        return harness.post_llm(
            self.base, self.api_key, self.model, messages,
            orch=self.orchestrator, timeout=timeout,
        )

    async def run(self, message: str, instructions: str | None = None,
                  on_event=None):
        import asyncio

        def _emit(ev: dict) -> None:
            if on_event is None:
                return
            try:
                on_event(ev)
            except Exception:
                pass

        sys_prompt = _SYSTEM_PROMPT + _catalog()
        if instructions:
            sys_prompt += "\n" + instructions
        try:
            from .microapps import looks_like_generate, skill_prompt

            if looks_like_generate(message):
                sys_prompt += "\n" + skill_prompt()
        except Exception:
            pass
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": message},
        ]
        used: list[str] = []
        last_result = None
        text = ""
        repaired = False
        cap = _round_cap()
        round_n = 0
        # FABRIC_NGRAM_LIVE=1: activate this lane's session buffer so the
        # first prefill of the run already sees the rolling state.
        live = False
        try:
            from . import latent as _latent
            live = _latent._ngram_live_enabled()
            if live:
                _latent._PLE_BUFFER.activate(_latent.live_session())
        except Exception:  # noqa: BLE001
            live = False
        for _ in range(cap):
            round_n += 1
            _emit({"t": "llm", "round": round_n})
            budget = _char_budget()
            if _history_chars(messages) > budget:
                note = _ngram_transfer_elided(messages)
                if note is not None:
                    _emit({"t": "ngram", **note})
                messages = _fit_history(messages, budget)
                _emit({"t": "ctx", "chars": _history_chars(messages)})
            try:
                text = await asyncio.to_thread(self._complete, messages)
            except RuntimeError as e:
                if not _is_ctx_error(str(e)):
                    raise
                half = max(12_000, budget // 2)
                note = _ngram_transfer_elided(messages)
                if note is not None:
                    _emit({"t": "ngram", "retry": True,
                           **note})
                messages = _fit_history(messages, half)
                _emit({"t": "ctx", "chars": _history_chars(messages),
                       "retry": True})
                text = await asyncio.to_thread(self._complete, messages)
            call = parse_tool_call(text)
            if call is None and not repaired and _dangling_tool(text):
                repaired = True
                _emit({"t": "repair"})
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Your reply was cut off before the closing "
                               "tag. Re-send the one complete <tool> block, "
                               "nothing else.",
                })
                continue
            if call is None:
                break
            name, args = call
            used.append(name)
            try:
                arg_blob = json.dumps(args, default=str)
            except Exception:
                arg_blob = str(args)[:300]
            _emit({"t": "tool", "name": name,
                   "args": arg_blob[:300]})
            t0 = time.monotonic()
            result = _dispatch(name, args)
            ms = int((time.monotonic() - t0) * 1000)
            ok = not (isinstance(result, dict) and result.get("error"))
            _emit({"t": "result", "name": name, "ok": ok, "ms": ms,
                   "preview": _trim(result, 240)})
            last_result = result
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Tool {name} returned:\n{_trim(result)}\n"
                           "Give the final answer, or call another tool.",
            })
        else:
            _emit({"t": "capped", "rounds": cap})
            out_text = _strip_think(text) or "(empty reply)"
            marker = (f"\n\n[duty agent hit the {cap}-leg tool limit "
                      "mid-task; partial results only]")
            if parse_tool_call(out_text) is not None or _dangling_tool(out_text):
                out_text = out_text.rstrip() + marker
            else:
                out_text += marker
            if live:
                _ngram_live_feed(message, out_text, _emit)
            return DutyResult(out_text, used, last_result=last_result)
        if live:
            _ngram_live_feed(message, _strip_think(text), _emit)
        return DutyResult(_strip_think(text) or "(empty reply)", used,
                          last_result=last_result)


def build_agent(model_name: str | None = None, extra_instructions: str = ""):
    # extra_instructions kept for call-site compat; per-run `instructions=`
    # on DutyAgent.run is the live path.
    _ = extra_instructions
    from . import harness

    orch = harness.orchestrator()
    if orch.get("external") and not orch.get("ready"):
        raise RuntimeError(
            f"{orch['label']} has no API key — set it in /harness")
    return DutyAgent(
        model_name=model_name or orch.get("model"),
        base=orch.get("base"),
        api_key=orch.get("api_key"),
        orchestrator=orch.get("id") or "local",
    )
