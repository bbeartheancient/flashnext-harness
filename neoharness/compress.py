"""Deterministic compaction of retired spans. No model, no VRAM, no latency.

Measured against the alternatives on the same input (docs/ngram-gate-plan.md §11):
this beat or matched a 4B compactor at every compression ratio tested, and the 4B
expanded its input (0.78-0.86x) on four of four prompts. The 4B was reliable only
under one narrow prompt shape and was given up: it held 7 GiB of VRAM to be
unreliable, which is the wrong trade for a lane that runs on every retirement.

Unit of selection is the CLAUSE, not the sentence. Sentence-level selection lost
8 of 15 facts at 3.35x, because one sentence routinely carries three of them
("raise to 400ms fixed staging, but the team error budget is 3% (now 3.4% weekly)")
and dropping the sentence drops all three.

This is a LOSSY side-channel, never the stored copy: recall() keeps searching the
original text, so whatever this drops is still recoverable.
"""
from __future__ import annotations

import os
import re

# Fraction of the original a compacted span is allowed to occupy.
RATIO = float(os.environ.get("NEOH_COMPACT_RATIO", "0.40"))
# Below this many chars a span is not worth compressing: the machinery costs more
# than it saves and the span is already small enough for the H_t block.
MIN_CHARS = int(os.environ.get("NEOH_COMPACT_MIN_CHARS", "240"))

# Load-bearing tokens: numbers, CamelCase, SCREAMING_SNAKE, `key:value`, bare a:b.
HARD = re.compile(r"\d|[A-Z][a-z]+[A-Z][A-Za-z]*|\b[A-Z]{2,}(?:_[A-Z]+)+\b|`[^`]+`|\b[a-z]+:[a-z]")
DATE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
# Conversational scaffolding that carries no fact.
HEDGE = re.compile(r"^(ok|okay|ok sure|sure|yes|no|thanks|thank you|got it|sounds good|"
                   r"what did we decide|and the|who owns|do we need|what is the rollback)\b", re.I)
# Clause boundaries: sentence ends, coordinating conjunctions, and ", X is/was ...".
CLAUSE = re.compile(r"(?<=[.;])\s+|\s+(?:but|so|and then|and also)\s+|"
                    r",\s+(?=[a-z]+ (?:is|was|are|were)\b)", re.I)
GATE = re.compile(r"\[gate:[^\]]*\]")
ROLE = re.compile(r"\[(?:user|assistant)\]\s*")


def _clauses(text: str) -> list[str]:
    out = []
    for line in GATE.sub("", text).replace("\n", " ").split(". "):
        for c in CLAUSE.split(line):
            c = ROLE.sub("", c).strip(" .;,")
            if len(c) > 8 and not HEDGE.match(c):
                out.append(c)
    return out


def _score(c: str) -> float:
    return (2.0 * len(HARD.findall(c))
            + 0.5 * len(DATE.findall(c))
            + 0.3 * len(re.findall(r"\b(?:19|20)\d{2}\b", c)))


def compact(text: str, ratio: float = RATIO) -> str:
    """The fact-bearing clauses of `text`, in reading order, within `ratio` of its length.

    Falls back to the input when compression would not pay. Read the result as a
    FACT INDEX, not a readable summary: at aggressive ratios it is fragment soup,
    which is fine for recall and bad for reading.
    """
    if len(text) < MIN_CHARS:
        return text
    clauses = _clauses(text)
    if len(clauses) < 2:
        return text
    budget = int(len(text) * ratio)
    ranked = sorted(range(len(clauses)), key=lambda i: (-_score(clauses[i]), i))
    kept, used = [], 0
    for i in ranked:
        if used + len(clauses[i]) > budget:
            continue
        kept.append(i)
        used += len(clauses[i])
    if not kept:
        return text
    kept.sort()                                   # restore reading order
    out = ". ".join(clauses[i] for i in kept)
    return out if 0 < len(out) < len(text) else text


def ratio_of(original: str, compacted: str) -> float:
    return len(original) / max(1, len(compacted))
