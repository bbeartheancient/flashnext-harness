"""Epoch-boundary experience consolidation (EvoHarness-RL §2.3).

Within an episode the policy writes insights to the ExperienceStore note
buffer via note[...]. At epoch boundaries the notes plus a trajectory
summary are merged into the main skill bank through add / update / remove
operations — "harness evolution": the bank expands early, then stabilizes
into a compact task-adaptive substrate (frequently recalled knowledge is
preserved, redundant entries merged, rarely useful skills evicted).

v1 uses deterministic, lexical consolidation (no LLM call): duplicate notes
are merged onto the best-matching existing skill (bumping its usage count),
genuinely new notes become new skills, and stale low-usage skills can be
pruned. The operation log is returned so a future consolidation model can be
dropped in behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bpe import ExperienceStore

MERGE_OVERLAP = 3  # shared content words to consider a note redundant


@dataclass
class ConsolidationResult:
    added: list[str] = field(default_factory=list)    # skill_ids
    updated: list[str] = field(default_factory=list)  # skill_ids
    removed: list[str] = field(default_factory=list)  # skill_ids
    notes_consumed: int = 0


def _content_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if len(w) > 3}


def _best_match(store: ExperienceStore, text: str, kind: str) -> dict | None:
    words = _content_words(text)
    if not words:
        return None
    best, best_overlap = None, 0
    for entry in store.skills.values():
        if entry["kind"] != kind:
            continue
        overlap = len(words & _content_words(entry["title"] + " " + entry["text"]))
        if overlap > best_overlap:
            best, best_overlap = entry, overlap
    return best if best_overlap >= MERGE_OVERLAP else None


def consolidate(
    store: ExperienceStore,
    trajectory_summary: str | None = None,
    prune_below_usage: int = 0,
) -> ConsolidationResult:
    """Merge the note buffer into the skill bank; optionally prune.

    - A note that lexically matches an existing same-kind skill UPDATES that
      skill (usage bump + text refresh) instead of duplicating it.
    - A novel note is ADDED as a new skill.
    - With ``prune_below_usage`` > 0, skills with usage_count below the
      threshold are REMOVED (forgetting); pinned by usage above it.
    """
    result = ConsolidationResult()
    for note in store.note_buffer:
        match = _best_match(store, note["text"], note["kind"])
        if match is not None:
            store.update(match["skill_id"], text=note["text"])
            match["usage_count"] += 1
            result.updated.append(match["skill_id"])
        else:
            title = " ".join(note["text"].split()[:6])
            entry = store.add(note["kind"], title, note["text"])
            result.added.append(entry["skill_id"])
        result.notes_consumed += 1
    store.note_buffer.clear()

    if trajectory_summary:
        words = _content_words(trajectory_summary)
        if words:
            title = " ".join(trajectory_summary.split()[:6])
            entry = store.add("task", title, trajectory_summary)
            result.added.append(entry["skill_id"])

    if prune_below_usage > 0:
        for sid, entry in list(store.skills.items()):
            if entry["usage_count"] < prune_below_usage:
                store.remove(sid)
                result.removed.append(sid)

    return result
