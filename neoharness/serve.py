"""OpenAI-compatible server: one endpoint for OpenCode and Hermes.

    python -m neoharness serve [--port 8393] [--workspace DIR] [--latent]

POST /v1/chat/completions runs the request's last user message as a
NeoHarness goal through GoalRunner (tool loop + BPE state + XYZ latent
lane when enabled) and returns the result as an ordinary chat completion.
Both stream=true (SSE step deltas) and non-streaming are supported.

Integration:
- OpenCode: provider baseURL http://127.0.0.1:8393/v1, model "neoharness"
- Hermes:  `hermes model` -> custom OpenAI-compatible endpoint, same URL

BPE state, skills, and trajectories persist in NEOH_STATE_DIR across
requests, so cross-session recall works for both clients. Goals run
serially (the duty GPU is serial anyway).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import Agent
from .bpe import BPEState
from .goals import GoalRunner
from .harness import Harness
from .skills import SkillStore
from .tools import make_core_tools

MODEL_ID = "neoharness-duty"


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, llm, workspace: Path, latent: bool,
                 max_steps: int, max_tokens: int):
        super().__init__(addr, _Handler)
        self.llm = llm
        self.workspace = workspace
        self.latent = latent
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.state = BPEState.load()
        self.goal_lock = threading.Lock()  # one goal at a time


def _extract_goal(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):  # OpenAI content parts
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text")
            return content or ""
    return ""


def _format_reply(result) -> str:
    ep = result.episode
    lines = []
    done = next((s.detail["done"] for s in reversed(ep.steps) if "done" in s.detail),
                None)
    if done:
        lines.append(done)
    elif ep.steps:
        lines.append(ep.steps[-1].observation or ep.steps[-1].action_text[:500])
    lines.append("")
    lines.append(f"[neoharness: solved={result.solved} steps={len(ep.steps)} "
                 f"tools={ep.env_calls} harness={ep.harness_calls}"
                 + (f" skill={result.skill_path}" if result.skill_path else "")
                 + "]")
    return "\n".join(lines)


def _chunk(model: str, content: str, finish: str | None = None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "delta": {"content": content} if content else {},
                     "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


class _Handler(BaseHTTPRequestHandler):
    server_version = "NeoHarness/0.3"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; keep stdout for step stream
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path in ("/ui", "/"):
            from .webui import PAGE
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model",
                "created": int(time.time()), "owned_by": "neoharness",
            }]})
        elif path == "/api/state":
            srv = self.server
            self._send_json(200, {
                "render": srv.state.render(),
                "belief_facts": len(srv.state.belief.facts),
                "progress_entries": len(srv.state.progress.entries),
                "experience_skills": len(srv.state.experience.skills),
            })
        elif path == "/api/skills":
            self._send_json(200, {"skills": SkillStore(self.server.state).list()})
        elif path in ("/health", "/v1/health"):
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        goal = _extract_goal(req.get("messages", []))
        if not goal:
            self._send_json(400, {"error": "no user message in request"})
            return
        stream = bool(req.get("stream"))
        srv = self.server

        events: queue.Queue = queue.Queue()

        def on_step(step):
            call = step.verb + (f" {step.arg}" if step.arg else "")
            events.put(f"> t{step.t} [{step.kind}] {call[:200]}\n")

        def run_goal():
            with srv.goal_lock:
                harness = Harness(srv.llm, state=srv.state)
                if srv.latent:
                    from .vectors import XYZBuffer
                    harness.buffer = XYZBuffer()
                agent = Agent(srv.llm, harness=harness,
                              tools=make_core_tools(srv.workspace),
                              max_steps=srv.max_steps)
                agent.on_step = on_step
                runner = GoalRunner(agent, SkillStore(harness.state))
                try:
                    return runner.run_goal(goal)
                finally:
                    harness.state.save()

        if not stream:
            try:
                result = run_goal()
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"goal failed: {exc}"})
                return
            self._send_json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", MODEL_ID),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": _format_reply(result)}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            })
            return

        # SSE streaming: run the goal in a thread, stream step deltas.
        holder: dict = {}

        def worker():
            try:
                holder["result"] = run_goal()
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            finally:
                events.put(None)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # no length; close = end of stream
        self.end_headers()
        threading.Thread(target=worker, daemon=True).start()
        model = req.get("model", MODEL_ID)
        try:
            while True:
                try:
                    item = events.get(timeout=20)
                except queue.Empty:
                    # First turn can take minutes on a busy engine; keep the
                    # connection (and the user's patience) alive.
                    self.wfile.write(b": waiting for the duty model...\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                self.wfile.write(_chunk(model, item).encode())
                self.wfile.flush()
            if "error" in holder:
                self.wfile.write(_chunk(model, f"\n[neoharness error: {holder['error']}]").encode())
            else:
                self.wfile.write(_chunk(model, "\n" + _format_reply(holder["result"])).encode())
            self.wfile.write(_chunk(model, "", finish="stop").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected mid-goal; the goal keeps running
        finally:
            self.close_connection = True


def make_server(host: str, port: int, llm, workspace: Path,
                latent: bool = False, max_steps: int = 40,
                max_tokens: int = 1024) -> _Server:
    return _Server((host, port), llm, workspace, latent, max_steps, max_tokens)


def main(argv: list[str] | None = None, llm=None) -> int:
    p = argparse.ArgumentParser(prog="neoharness serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8393)
    p.add_argument("--workspace", default=".")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--latent", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    args = p.parse_args(argv)

    import os
    if args.latent or os.environ.get("NEOH_LATENT") == "1":
        os.environ["NEOH_LATENT"] = "1"
        args.latent = True
    if llm is None:
        from .llm import DutyLLM
        llm = DutyLLM(base_url=args.base_url, model=args.model,
                      max_tokens=args.max_tokens)

    server = make_server(args.host, args.port, llm,
                         Path(args.workspace).resolve(), args.latent,
                         args.max_steps, args.max_tokens)
    print(f"neoharness serving on http://{args.host}:{args.port}/v1 "
          f"(workspace={server.workspace}, latent={'on' if args.latent else 'off'})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.state.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
