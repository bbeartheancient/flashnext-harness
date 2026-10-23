"""Deterministic fan-out workers: search without a language model.

Why this exists. Measured on the live Strata service, four LLM subagents took
13.6 s and 17 model calls, and TWO of the four returned empty - a 125B model
emitting prose instead of a <tool> block. Concurrency was 0.82x serial, because
one decode stream already saturates the GPU and each worker's tool loop is
sequential.

But look at what the workers that worked actually did: list_dir and run_shell.
Every job was a syscall. So the subagents do not need parameters; they need to
SEARCH. This module does that with no model at all:

    terms(sub-goal) -> grep sweep -> read the strongest matches -> report

It reuses the same jailed ToolRegistry the agent uses, so the sandbox and the
allowlist are unchanged. It never invents: every line it reports came out of a
file. When a search finds nothing it says so, rather than returning empty -
which is the failure mode that made the LLM version unreliable.
"""

from __future__ import annotations

import re
from pathlib import Path

from .handoff import BUDGET
from .tools import workspace_root

# words that carry no search signal in a task description
STOP = frozenset("""
a an the and or but of to in on at by for with without from into onto over under
between within across through during before after above below up down out off
is are was were be been being am do does did doing done have has had having
will would shall should can could may might must let
this that these those there here it its it's they them their he she his her
you your we our i me my mine us not no nor so such than then too very
use used using uses find show list read write create make get give tell explain
describe name names named call called return returns define defines defined
check checks test tests run runs see look looking look up want need
every any all some more most other another each per both few many much
file files code function functions class classes module modules project projects
workspace goal task work works about also just only very much real exact string
text name value values key keys line lines file file's contents content
after before while during
""".split())

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def terms(text: str, limit: int = 6) -> list[str]:
    """Salient search terms from a sub-goal: identifiers first, then long words.

    Identifiers (alpha_main, ffn_gate) are the strongest signal in a codebase
    goal, so they rank above prose."""
    ids, words = [], []
    for w in WORD_RE.findall(text or ""):
        if w.lower() in STOP:
            continue
        if "_" in w or (w[:1].isupper() and w[1:].islower()):
            ids.append(w)
        elif len(w) >= 4:
            words.append(w)
    seen, out = set(), []
    for w in ids + words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _safe(pattern: str) -> str | None:
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        return None


class Searcher:
    """One fan-out worker. Deterministic: same workspace + same goal -> same report."""

    def __init__(self, tools, workspace=None, max_matches: int = 25,
                 max_files: int = 4, max_chars: int | None = None, fanout: int = 1):
        self.tools = tools
        self.workspace = Path(workspace) if workspace else None
        self.max_matches = max_matches
        self.max_files = max_files
        # A worker may not claim more than its share of the handoff budget, or N
        # of them overflow it and the CONTRACT stage drops the oldest (measured:
        # 4 redundant workers, 16000-char budget, worker 1 lost).
        self.fanout = max(1, fanout)
        self.max_chars = max_chars or (BUDGET // self.fanout)

    # -- primitives, all through the jailed registry --------------------------
    def _call(self, name: str, **args) -> str:
        try:
            return self.tools.call(name, args)
        except Exception as exc:                      # noqa: BLE001 - fail soft
            return f"({name} failed: {type(exc).__name__}: {exc})"

    def _tree(self, root: str = ".", depth: int = 3, cap: int = 28) -> list[str]:
        """A bounded recursive listing. Structure-only reporting depends on this."""
        lines: list[str] = []
        # self.workspace (the agent's sandbox) wins; else the registry's own root.
        # "." is resolved against it, so a test's tmp_path stays the tree root.
        anchor = self.workspace or workspace_root()
        try:
            base = (anchor / root).resolve() if not Path(root).is_absolute() else Path(root).resolve()
            rootd = anchor.resolve()
            if base != rootd and rootd in base.parents:
                base = rootd
        except OSError:
            return [f"(cannot read {root})"]

        def walk(d: Path, level: int):
            if len(lines) >= cap:
                return
            if level > depth:
                lines.append("  " * level + "...")
                return
            try:
                entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
            except OSError:
                return
            for e in entries[: max(1, cap - len(lines))]:
                if len(lines) >= cap:
                    return
                rel = e.relative_to(base)
                if e.name in (".git", "__pycache__", ".venv", "node_modules",
                              ".mypy_cache", ".pytest_cache", "dist", "build"):
                    continue                      # internals are not structure
                if e.is_dir():
                    lines.append("  " * level + f"{rel}/")
                    walk(e, level + 1)
                else:
                    try:
                        n = e.stat().st_size
                    except OSError:
                        n = -1
                    lines.append("  " * level + f"{rel} ({n} B)")
        walk(base, 0)
        return lines or ["(empty workspace)"]

    def _fit(self, text: str) -> str:
        """Enforce the per-worker share of the handoff budget, saying what was cut
        (law hand_cut_says: a cut item says by how much)."""
        if len(text) <= self.max_chars:
            return text
        cut = self.max_chars - 48
        return text[:cut] + f"\n    ...[{len(text) - cut} more characters]"

    def search(self, sub_goal: str, context: str = "") -> str:
        out: list[str] = []
        ts = terms(sub_goal)
        out.append(f"terms: {', '.join(ts) if ts else '(none)'}")
        out.append("")

        # 1. STRUCTURE FIRST, and always. This is the worker's reliable
        #    backbone: whatever the terms find, the tree is true and present, so
        #    a worker never comes back empty (the LLM workers' failure mode).
        out.append("structure:")
        out += ["  " + ln for ln in self._tree()]
        out.append("")

        # 2. grep sweep over the salient terms
        found: dict[str, list[str]] = {}
        for t in ts:
            pat = _safe(re.escape(t))
            if pat is None:
                continue
            res = self._call("grep", pattern=pat, path=".", max_matches=self.max_matches)
            if res and not res.startswith("(no matches)") and "(" not in res[:1]:
                hits = [h for h in res.splitlines() if h.strip()]
                if hits:
                    found[t] = hits
        if not found:
            # structure-only: no term matched, but the tree above is still the
            # most useful true thing this worker can hand back.
            out.append("grep: no term matched - structure-only report")
            return self._fit("\n".join(out))
        out.append("grep matches:")
        for t, hits in found.items():
            out.append(f"  [{t}] {len(hits)} hit(s)")
            out += [f"    {h}" for h in hits[:8]]
        out.append("")

        # 3. read the files that matched most strongly
        scored: list[tuple[int, str]] = []
        for hits in found.values():
            for h in hits:
                m = re.match(r"([^:]+):", h)
                if m:
                    scored.append((1, m.group(1)))
        rank: dict[str, int] = {}
        for _s, f in scored:
            rank[f] = rank.get(f, 0) + 1
        top = sorted(rank.items(), key=lambda kv: -kv[1])[: self.max_files]
        if top:
            out.append("strongest files:")
            for f, n in top:
                body = self._call("read_file", path=f, offset=0, limit=12)
                out.append(f"  --- {f} ({n} hits) ---")
                out += [f"    {ln}" for ln in body.splitlines()[:12]]

        return self._fit("\n".join(out))
