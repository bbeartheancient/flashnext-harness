"""Repo vocabulary index — symbol/heading -> file:line.

The duty agent burns legs on ls/grep/read probes every time it needs to
locate something. This module pre-parses the working tree once (TTL
refresh) into {token: [(path, line, kind)]} so lookups are one tool call:

    code_index("post_llm")  ->  fabric/harness.py:512 def post_llm

Python uses ast (defs/classes/UPPER consts); markdown indexes headings;
C/C++ gets class/struct/function regexes. Vendor/build trees are skipped.
"""

from __future__ import annotations

import ast
import os
import re
import threading
import time
from pathlib import Path

_TTL_S = 30.0
_LIMIT = 24
_LOCK = threading.Lock()
_CACHE: dict = {"built": 0.0, "index": {}, "files": 0}

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", ".venv-fabric",
    ".venv-vllm", "build", "build-comfy", "build-onednn", "cache",
    "third_party", "vendor", "logs", ".grok", "LCARS-26", "LCARSvar1",
    "LCARSvar2", "Qwen25VL",
}
_SUFFIXES = {".py", ".md", ".hpp", ".h", ".cpp", ".cc", ".c"}
_SKIP_SUFFIX = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".woff", ".woff2",
    ".so", ".a", ".o", ".bin", ".pt", ".safetensors", ".gguf",
    ".zip", ".tar", ".gz", ".mp4", ".pyc", ".whl", ".json",
}


def _root() -> Path:
    return Path(os.environ.get(
        "FABRIC_PI_CWD",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )).resolve()


_MD_HEAD = re.compile(r"^#{1,4}\s+(.+?)\s*#*\s*$", re.M)
_C_TYPE = re.compile(r"^\s*(?:class|struct|namespace)\s+([A-Za-z_]\w*)", re.M)
_C_FUNC = re.compile(
    r"^[ \t]*(?:[A-Za-z_][\w \t\*&:<>,]*?)?\b([A-Za-z_]\w*)\s*"
    r"\([^;{}]*\)\s*(?:const\s*)?\{", re.M)


def _walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".venv")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in _SUFFIXES and p.suffix.lower() not in _SKIP_SUFFIX:
                yield p


def _py_entries(text: str):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node.lineno, "def"
        elif isinstance(node, ast.ClassDef):
            yield node.name, node.lineno, "class"
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Name) and len(t.id) > 2
                        and t.id.isupper()):
                    yield t.id, node.lineno, "const"


def _file_entries(path: Path, rel: str):
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if suffix == ".py":
        for name, line, kind in _py_entries(text):
            yield name, line, kind
    elif suffix == ".md":
        for i, line_text in enumerate(text.splitlines(), 1):
            m = _MD_HEAD.match(line_text)
            if m and len(m.group(1)) > 2:
                yield m.group(1).strip(), i, "md"
    else:
        for m in _C_TYPE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            yield m.group(1), line, "type"
        for m in _C_FUNC.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            yield m.group(1), line, "def"


def _scan() -> dict:
    index: dict[str, list] = {}
    files = 0
    root = _root()
    for path in _walk_files(root):
        files += 1
        rel = str(path.relative_to(root))
        for name, line, kind in _file_entries(path, rel):
            key = name.strip().lower()
            if not key:
                continue
            index.setdefault(key, []).append((rel, line, kind, name.strip()))
    return {"index": index, "files": files, "built": time.monotonic()}


def _get(force: bool = False) -> dict:
    with _LOCK:
        if force or not _CACHE["index"] \
                or time.monotonic() - _CACHE["built"] > _TTL_S:
            fresh = _scan()
            _CACHE["built"] = fresh["built"]
            _CACHE["index"] = fresh["index"]
            _CACHE["files"] = fresh["files"]
        return _CACHE


def reset_cache() -> None:
    with _LOCK:
        _CACHE["built"] = 0.0
        _CACHE["index"] = {}
        _CACHE["files"] = 0


def stats() -> dict:
    c = _get()
    return {"files": c["files"], "symbols": sum(len(v) for v in
                                                c["index"].values()),
            "age_s": round(time.monotonic() - c["built"], 1),
            "root": str(_root())}


def find(query: str, limit: int = _LIMIT) -> list[dict]:
    """Exact -> prefix -> substring match over the lowercase vocabulary."""
    q = (query or "").strip().lower()
    if not q:
        return []
    idx = _get()["index"]
    exact = idx.get(q, [])
    rows = list(exact)
    seen = set(exact)
    for key, hits in sorted(idx.items()):
        if len(rows) >= limit * 3:
            break
        if key.startswith(q) or q in key:
            for h in hits:
                if h not in seen:
                    rows.append(h)
                    seen.add(h)
    rows.sort(key=lambda r: (r[0], r[1]))
    return [{"file": f, "line": ln, "kind": k, "name": n}
            for f, ln, k, n in rows[:limit]]
