"""BPE harness state: Belief, Progress, Experience stores (EvoHarness-RL §2.1).

Text-first, policy-facing state. Each store renders a compact text view that
is spliced into the duty model's prompt as H_t = (B_t, P_t, E_t). The same
rendered text is what the XYZ vector lane (vectors.py) hidden-captures into
the n-gram buffer, so the text view and the latent coordinates never diverge.

Persistence: JSON under NEOH_STATE_DIR (default ~/.local/state/neoharness).
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path

# One re-entrant lock guards every store container. The 1:4 setup (four agents
# running against one duty server) shares a single BPEState across goals, so
# read-modify-write sequences like update()'s evict, commit()'s append+evict and
# recall()'s usage_count bump are no longer atomic on their own. The critical
# sections are in-memory list/dict work measured in microseconds, so a single
# coarse lock costs nothing next to a model step; os.replace in save() already
# keeps the on-disk file whole.
_STATE_LOCK = threading.RLock()


def _guarded(fn):
    """Serialize a store method against the shared _STATE_LOCK (re-entrant)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _STATE_LOCK:
            return fn(*args, **kwargs)

    return wrapper


DEFAULT_MAX_FACTS = 256
DEFAULT_MAX_SUBGOALS = 64
DEFAULT_MAX_SKILLS = 512
RECALL_TOP_K = 4

SKILL_KINDS = ("general", "task", "mistake", "prior")

PROGRESS_STATUSES = ("pending", "attempted", "done", "blocked")


def state_dir() -> Path:
    d = Path(
        os.environ.get("NEOH_STATE_DIR", "~/.local/state/neoharness")
    ).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


class BeliefStore:
    """Environment estimate: task-relevant facts about objects.

    facts: object name -> {"state": str, "location": str,
                           "relations": [str, ...], "updated": epoch}
    """

    def __init__(self, max_facts: int = DEFAULT_MAX_FACTS):
        self.max_facts = max_facts
        self.facts: dict[str, dict] = {}

    @_guarded
    def update(
        self,
        obj: str,
        state: str | None = None,
        location: str | None = None,
        relations: list[str] | None = None,
    ) -> dict:
        obj = obj.strip().lower()
        if not obj:
            raise ValueError("belief object name is empty")
        entry = self.facts.setdefault(
            obj, {"state": "", "location": "", "relations": [], "updated": 0.0}
        )
        if state is not None:
            entry["state"] = state
        if location is not None:
            entry["location"] = location
        if relations is not None:
            entry["relations"] = list(relations)
        entry["updated"] = time.time()
        if len(self.facts) > self.max_facts:
            # Evict the stalest fact (LRU by last update).
            stalest = min(self.facts, key=lambda k: self.facts[k]["updated"])
            if stalest != obj:
                del self.facts[stalest]
        return entry

    @_guarded
    def track(self, query: str = "world") -> str:
        """track[object] -> one fact view; track[world] -> compact summary."""
        query = query.strip().lower()
        if query in ("", "world"):
            return self.render()
        entry = self.facts.get(query)
        if entry is None:
            return f"{query}: unknown"
        return self._render_entry(query, entry)

    def _render_entry(self, obj: str, entry: dict) -> str:
        parts = [obj]
        if entry["state"]:
            parts.append(f"state={entry['state']}")
        if entry["location"]:
            parts.append(f"at={entry['location']}")
        if entry["relations"]:
            parts.append("rel=[" + "; ".join(entry["relations"]) + "]")
        return " ".join(parts)

    @_guarded
    def render(self, limit: int = 32) -> str:
        if not self.facts:
            return "(belief empty)"
        recent = sorted(
            self.facts.items(), key=lambda kv: kv[1]["updated"], reverse=True
        )[:limit]
        return "\n".join(self._render_entry(o, e) for o, e in recent)

    # -- persistence --
    @_guarded
    def to_json(self) -> dict:
        return {"max_facts": self.max_facts, "facts": self.facts}

    @classmethod
    def from_json(cls, data: dict) -> "BeliefStore":
        s = cls(max_facts=data.get("max_facts", DEFAULT_MAX_FACTS))
        s.facts = {k: dict(v) for k, v in data.get("facts", {}).items()}
        return s


class ProgressStore:
    """Committed execution record: bounded subgoal-status list."""

    def __init__(self, max_subgoals: int = DEFAULT_MAX_SUBGOALS):
        self.max_subgoals = max_subgoals
        self.entries: list[dict] = []  # {"subgoal", "status", "updated"}

    @_guarded
    def commit(self, subgoal: str, status: str = "done") -> dict:
        subgoal = subgoal.strip()[:160]  # cap: models sometimes quote H_t back
        if not subgoal:
            raise ValueError("subgoal is empty")
        if status not in PROGRESS_STATUSES:
            raise ValueError(
                f"status must be one of {PROGRESS_STATUSES}, got {status!r}"
            )
        for entry in self.entries:
            if entry["subgoal"] == subgoal:
                entry["status"] = status
                entry["updated"] = time.time()
                return entry
        entry = {"subgoal": subgoal, "status": status, "updated": time.time()}
        self.entries.append(entry)
        if len(self.entries) > self.max_subgoals:
            # Drop the oldest completed entry first; never drop open work.
            done = [i for i, e in enumerate(self.entries) if e["status"] == "done"]
            del self.entries[done[0] if done else 0]
        return entry

    @_guarded
    def open_subgoals(self) -> list[dict]:
        return [e for e in self.entries if e["status"] in ("pending", "blocked")]

    @_guarded
    def render(self, limit: int = 16) -> str:
        if not self.entries:
            return "(progress empty)"
        recent = self.entries[-limit:]
        return "\n".join(f"[{e['status']}] {e['subgoal']}" for e in recent)

    @_guarded
    def to_json(self) -> dict:
        return {"max_subgoals": self.max_subgoals, "entries": self.entries}

    @classmethod
    def from_json(cls, data: dict) -> "ProgressStore":
        s = cls(max_subgoals=data.get("max_subgoals", DEFAULT_MAX_SUBGOALS))
        s.entries = [dict(e) for e in data.get("entries", [])]
        return s


class ExperienceStore:
    """Cross-episode skill bank: general / task / mistake / prior entries.

    recall[query] = lexical top-k with usage-count bump (paper §2.3).
    note[insight] = append to the session note buffer, merged into the bank
    at consolidation time (consolidate.py).
    """

    def __init__(self, max_skills: int = DEFAULT_MAX_SKILLS):
        self.max_skills = max_skills
        self.skills: dict[str, dict] = {}
        self.note_buffer: list[dict] = []
        self._seq = 0

    def _new_id(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}-{self._seq:04d}"

    @_guarded
    def add(self, kind: str, title: str, text: str, skill_id: str | None = None) -> dict:
        if kind not in SKILL_KINDS:
            raise ValueError(f"kind must be one of {SKILL_KINDS}, got {kind!r}")
        sid = skill_id or self._new_id(kind)
        entry = {
            "skill_id": sid,
            "kind": kind,
            "title": title.strip(),
            "text": text.strip(),
            "usage_count": 0,
            "updated": time.time(),
        }
        self.skills[sid] = entry
        self._evict_if_over()
        return entry

    @_guarded
    def update(self, skill_id: str, text: str | None = None, title: str | None = None) -> dict:
        entry = self.skills[skill_id]
        if text is not None:
            entry["text"] = text.strip()
        if title is not None:
            entry["title"] = title.strip()
        entry["updated"] = time.time()
        return entry

    @_guarded
    def remove(self, skill_id: str) -> None:
        self.skills.pop(skill_id, None)

    @_guarded
    def recall(self, query: str, k: int = RECALL_TOP_K) -> list[dict]:
        """Lexical overlap top-k; bumps usage_count of returned entries."""
        q = {w for w in query.lower().split() if len(w) > 2}
        scored = []
        for entry in self.skills.values():
            words = set((entry["title"] + " " + entry["text"]).lower().split())
            overlap = len(q & words)
            if overlap:
                scored.append((overlap, entry["usage_count"], entry))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        hits = [e for _, _, e in scored[:k]]
        for entry in hits:
            entry["usage_count"] += 1
        return hits

    @_guarded
    def note(self, insight: str, kind: str = "general") -> dict:
        if kind not in SKILL_KINDS:
            raise ValueError(f"kind must be one of {SKILL_KINDS}, got {kind!r}")
        note = {"kind": kind, "text": insight.strip()[:500], "noted": time.time()}
        self.note_buffer.append(note)
        return note

    def _evict_if_over(self) -> None:
        # Frequently recalled knowledge is preserved; rarely useful skills
        # are evicted first (paper: harness evolution).
        while len(self.skills) > self.max_skills:
            victim = min(
                self.skills.values(),
                key=lambda e: (e["usage_count"], e["updated"]),
            )
            del self.skills[victim["skill_id"]]

    @_guarded
    def render(self, limit: int = 12) -> str:
        if not self.skills:
            return "(experience empty)"
        top = sorted(
            self.skills.values(),
            key=lambda e: (e["usage_count"], e["updated"]),
            reverse=True,
        )[:limit]
        return "\n".join(f"({e['kind']}) {e['title']}: {e['text']}" for e in top)

    @_guarded
    def to_json(self) -> dict:
        return {
            "max_skills": self.max_skills,
            "skills": self.skills,
            "note_buffer": self.note_buffer,
            "seq": self._seq,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ExperienceStore":
        s = cls(max_skills=data.get("max_skills", DEFAULT_MAX_SKILLS))
        s.skills = {k: dict(v) for k, v in data.get("skills", {}).items()}
        s.note_buffer = [dict(n) for n in data.get("note_buffer", [])]
        s._seq = data.get("seq", len(s.skills))
        return s


class BPEState:
    """H_t = (B_t, P_t, E_t) with JSON persistence."""

    def __init__(
        self,
        belief: BeliefStore | None = None,
        progress: ProgressStore | None = None,
        experience: ExperienceStore | None = None,
    ):
        self.belief = belief or BeliefStore()
        self.progress = progress or ProgressStore()
        self.experience = experience or ExperienceStore()

    @_guarded
    def render(self) -> str:
        """The policy-facing harness view spliced into the prompt."""
        return (
            "### Belief\n" + self.belief.render() + "\n"
            "### Progress\n" + self.progress.render() + "\n"
            "### Experience\n" + self.experience.render()
        )

    # -- persistence --
    @_guarded
    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else state_dir() / "bpe_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "belief": self.belief.to_json(),
                    "progress": self.progress.to_json(),
                    "experience": self.experience.to_json(),
                },
                indent=1,
            )
        )
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BPEState":
        p = Path(path) if path else state_dir() / "bpe_state.json"
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            belief=BeliefStore.from_json(data.get("belief", {})),
            progress=ProgressStore.from_json(data.get("progress", {})),
            experience=ExperienceStore.from_json(data.get("experience", {})),
        )
