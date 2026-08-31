# SPDX-License-Identifier: Apache-2.0
"""BPE bridge for the OpenCode shim (NeoHarness state in OpenCode's loop).

OpenCode manages its own context and compacts client-side; when the
summary drops running state the session gets lost (measured: the WELDING
session ran 10.4M cumulative tokens without persisting one fact; the
NeoHarness long-horizon bench shows raw loops fail 0/3 under forced
elision where BPE externalization passes).  This bridge injects the BPE
harness into the shim's verbatim forward:

  - INJECT: a compact protocol cheat-sheet + the session's rendered
    H_t = (Belief, Progress, Experience) appended to the first system
    message of every non-compaction request (capped at
    FABRIC_OPENCODE_BPE_CHARS, default 1500).  After a compaction the
    next request gets a "context was compacted" banner plus any pending
    harness answers.
  - APPLY: <action>commit/note/track/recall</action> blocks in the
    assistant reply are dispatched against the per-session BPE state
    (persisted per session under NEOH_STATE_DIR) — the tags pass through
    to OpenCode harmlessly (rendered as text); the state they write is
    what survives the next compaction.
  - track/recall answers ride back in the NEXT request's injection
    (pending-answer queue), closing the read path.

State lives in NeoHarness (imported, not vendored: NEOH_ROOT env,
default $HOME/NeoHarness).  Fail-soft everywhere: if NeoHarness is
unavailable the bridge disables itself and the shim behaves exactly as
before.  Gate: FABRIC_OPENCODE_BPE=1 (drop-in 65-opencode-bpe.conf).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

_lock = threading.RLock()
_state_cache: dict[str, object] = {}
_pending: dict[str, list[str]] = {}      # session -> harness answers
_compacted: set[str] = set()             # sessions awaiting a banner
_nh = None                               # neoharness modules (lazily)

_ACTION_RE = re.compile(
    r"<action>\s*([A-Za-z_][\w -]*?)\s*(?:\[(.*?)\])?\s*</action>", re.S)
# Observed failure mode: the duty model drops the <action> wrapper and
# emits a bare line "commit[x | attempted]".  Line-anchored fallback,
# only used when no tagged action matched.
_BARE_ACTION_RE = re.compile(
    r"^\s*(commit|note|track|recall)\s*\[(.*?)\]\s*$", re.M)
_READ_VERBS = ("track", "recall")

_CHEATSHEET = """\
### Harness (persistent state across compaction)
You may add ONE harness action per reply, written EXACTLY like this —
the <action> tags are required:
  <action>commit[subgoal | pending|attempted|done|blocked]</action>
  <action>note[general|task|mistake|prior | insight]</action>
  <action>track[world]</action>  <action>recall[query]</action>
The action is recorded outside the transcript and SURVIVES context
compaction; track/recall answers arrive with the next turn.
After any compaction, re-orient from the injected state BEFORE continuing.
Commit running totals/progress to the harness or to the project's state
file as you go — never let verified facts live only in this conversation."""


def enabled() -> bool:
    if os.environ.get("FABRIC_OPENCODE_BPE", "0") != "1":
        return False
    return _load_nh() is not None


def _load_nh():
    global _nh
    if _nh is not None:
        return _nh or None
    root = os.environ.get("NEOH_ROOT", str(Path.home() / "NeoHarness"))
    try:
        if root not in sys.path:
            sys.path.insert(0, root)
        from neoharness import actions, bpe  # noqa: F401
        _nh = {"actions": actions, "bpe": bpe}
    except Exception:  # noqa: BLE001
        _nh = False
    return _nh or None


def _state(session: str):
    with _lock:
        st = _state_cache.get(session)
        if st is None:
            nh = _load_nh()
            path = Path(nh["bpe"].state_dir()) / f"opencode-{session}.json"
            st = nh["bpe"].BPEState.load(path)
            st._bridge_path = path
            _state_cache[session] = st
        return st


def _save(session: str) -> None:
    st = _state_cache.get(session)
    if st is not None:
        st.save(st._bridge_path)


def _cap(text: str) -> str:
    limit = int(os.environ.get("FABRIC_OPENCODE_BPE_CHARS", "1500"))
    return text if len(text) <= limit else text[:limit] + "\n...(state truncated)"


def mark_compaction(session: str) -> None:
    with _lock:
        _compacted.add(session)


def inject(body: dict, session: str) -> dict:
    """Return a copy of ``body`` with the BPE block merged into the first
    system message.  Never mutates ``body``."""
    nh = _load_nh()
    if nh is None:
        return body
    with _lock:
        st = _state(session)
        block = _CHEATSHEET + "\n\n### Harness state (H_t)\n" + _cap(st.render())
        if session in _compacted:
            _compacted.discard(session)
            block = ("[harness: the conversation was just compacted — "
                     "re-orient from this committed state]\n\n") + block
        answers = _pending.pop(session, [])
        if answers:
            block += "\n\n### Harness answers\n" + "\n".join(answers[:8])

        out = dict(body)
        msgs = list(out.get("messages") or [])
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                m = dict(m)
                c = m.get("content")
                if isinstance(c, list):
                    m["content"] = list(c) + [{"type": "text", "text": "\n\n" + block}]
                else:
                    m["content"] = str(c or "") + "\n\n" + block
                msgs[i] = m
                break
        else:
            msgs.insert(0, {"role": "system", "content": block})
        out["messages"] = msgs
        return out


def apply_reply(text: str, session: str) -> int:
    """Dispatch every <action> block in an assistant reply; returns the
    number applied.  Read-verb observations queue for the next inject."""
    nh = _load_nh()
    if nh is None or not text:
        return 0
    applied = 0
    with _lock:
        st = _state(session)
        found = _ACTION_RE.findall(text)
        if not found:
            found = [(v, a) for v, a in _BARE_ACTION_RE.findall(text)
                     if "]" not in a and "[" not in a]
        for verb, arg in found:
            tag = f"<action>{verb.strip()}[{arg or ''}]</action>"
            try:
                result = nh["actions"].dispatch(tag, st)
            except Exception:  # noqa: BLE001
                continue
            applied += 1
            if result.kind == "harness" and result.verb in _READ_VERBS:
                _pending.setdefault(session, []).append(
                    f"{result.verb}[{result.arg}] → "
                    f"{result.observation[:400]}")
        if applied:
            _save(session)
    return applied
