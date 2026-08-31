"""Tool protocol + core toolset (Hermes-style, sandboxed).

The duty 27B has no native tool_choice on this stack, so tools are invoked
prompt-side: the model emits ``<tool>name {"arg": ...}</tool>`` and agent.py
parses and executes it (same convention as M1 fabric/agent.py's <tool> loop).

Safety model (v1 = allowlist, Hermes's approval model is future work):
- File tools are jailed to NEOH_WORKSPACE (default: cwd). ``..`` escapes and
  absolute paths outside the root are rejected.
- run_shell only executes commands whose first word is in NEOH_SHELL_ALLOW
  (default: ls, cat, grep, find, echo, pwd). 30 s timeout, output capped.
- All tool output is capped at NEOH_TOOL_MAX_CHARS (default 4000).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

TOOL_MAX_CHARS = int(os.environ.get("NEOH_TOOL_MAX_CHARS", "4000"))
SHELL_TIMEOUT_S = 30
DEFAULT_SHELL_ALLOW = "ls,cat,grep,find,echo,pwd"


class ToolError(Exception):
    """Raised for sandbox violations and bad args; surfaced to the model."""


def workspace_root() -> Path:
    return Path(os.environ.get("NEOH_WORKSPACE", os.getcwd())).resolve()


def _jail(path: str, root: Path) -> Path:
    p = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if p != root and root not in p.parents:
        raise ToolError(f"path escapes workspace: {path!r}")
    return p


def _cap(text: str) -> str:
    if len(text) > TOOL_MAX_CHARS:
        return text[:TOOL_MAX_CHARS] + f"\n...(truncated at {TOOL_MAX_CHARS} chars)"
    return text


@dataclass
class Tool:
    name: str
    description: str
    args_spec: dict  # {arg_name: "type — description"}
    run: object  # callable(**kwargs) -> str

    def spec_text(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.args_spec.items())
        return f"{self.name}({args}) — {self.description}"

    def __call__(self, args: dict) -> str:
        try:
            # Tolerate the model wrapping args in a single {"arg": {...}}.
            if set(args) == {"arg"} and isinstance(args["arg"], dict):
                args = args["arg"]
            return _cap(str(self.run(**args)))
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"bad args for {self.name}: {exc}") from exc


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def call(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(
                f"unknown tool {name!r}; available: {', '.join(sorted(self.tools))}"
            )
        return tool(args)

    def spec_text(self) -> str:
        return "\n".join(t.spec_text() for t in self.tools.values())


# ---------------------------------------------------------------------------
# Core tools
# ---------------------------------------------------------------------------

def make_core_tools(root: Path | None = None) -> ToolRegistry:
    root = (root or workspace_root()).resolve()

    def read_file(path: str, offset: int = 0, limit: int = 400) -> str:
        p = _jail(path, root)
        if not p.is_file():
            raise ToolError(f"not a file: {path!r}")
        lines = p.read_text(errors="replace").splitlines()
        return "\n".join(lines[offset : offset + limit])

    def write_file(path: str, content: str, append: bool = False) -> str:
        p = _jail(path, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with p.open("a") as fh:
                fh.write(content)
        else:
            p.write_text(content)
        return f"wrote {len(content)} chars to {p.relative_to(root)}"

    def list_dir(path: str = ".") -> str:
        p = _jail(path, root)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path!r}")
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return "\n".join(entries) or "(empty)"

    def grep(pattern: str, path: str = ".", max_matches: int = 50) -> str:
        p = _jail(path, root)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad regex: {exc}") from exc
        hits = []
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(root)}:{i}: {line}")
                        if len(hits) >= max_matches:
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) or "(no matches)"

    def run_shell(command: str) -> str:
        allow = {
            w.strip()
            for w in os.environ.get("NEOH_SHELL_ALLOW", DEFAULT_SHELL_ALLOW).split(",")
            if w.strip()
        }
        first = command.strip().split(None, 1)[0] if command.strip() else ""
        if first not in allow:
            raise ToolError(
                f"command {first!r} not in NEOH_SHELL_ALLOW ({', '.join(sorted(allow))})"
            )
        try:
            proc = subprocess.run(
                command, shell=True, cwd=root, capture_output=True, text=True,
                timeout=SHELL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {SHELL_TIMEOUT_S}s") from None
        out = proc.stdout + proc.stderr
        return f"(exit {proc.returncode})\n{out.strip() or '(no output)'}"

    return ToolRegistry([
        Tool("read_file", "read lines from a workspace file",
             {"path": "str", "offset": "int (0-based line, optional)",
              "limit": "int (max lines, optional)"}, read_file),
        Tool("write_file", "write (or append) a workspace file",
             {"path": "str", "content": "str",
              "append": "bool (optional)"}, write_file),
        Tool("list_dir", "list a workspace directory",
             {"path": "str (optional)"}, list_dir),
        Tool("grep", "regex search across workspace files",
             {"pattern": "regex", "path": "str (optional)",
              "max_matches": "int (optional)"}, grep),
        Tool("run_shell", "run an allowlisted shell command in the workspace",
             {"command": "str"}, run_shell),
    ])
