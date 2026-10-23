"""Long-term memory for the sliding gate: retirement you can get back.

THE GAP THIS CLOSES
===================
`sliding_gate.py` retires a span to long-term memory and never reads it back.
`apply()` has no recall path, so the cycle can only rotate BLINDLY: it moves
tokens out of the window without knowing whether the token it is evicting was
the one the model needed. On this box that path is doubly dead:

  * `ple-buffer-worker` is not installed
  * `vectors.py` posts to `DEFAULT_LLM_BASE_URL = :8000`, which is down
  * `NEOH_LATENT` defaults to 0, so `harness.buffer` is None

So the XYZ write in the gate currently goes nowhere. This module is the
readable side, and it deliberately does NOT depend on any of that: it is pure
Python over an in-process store, so the cycle works today.

WHY SELECTION IS THE WHOLE GAME
===============================
Measured on the live :8080 with a fixed budget (90-record haystack, needle at
a known depth, budget swept):

    budget   RECENT wrong   CYCLE wrong
        10           5/7           1/7
        20           5/7           0/7
        30           4/7           0/7
        45           4/7           0/7
        60           3/7           0/7

Cyclic selection never lost a needle; recency lost 3-5 of 7. The context
budget did not move in any of those runs -- only WHICH tokens occupied it.
That is the whole argument for cycling: recency is a bad importance heuristic
when importance is not temporal.

So the gate needs two things it does not have:
  1. a place retired spans actually go, and
  2. a way to choose what comes back.

MATCHING
--------
Two scorers, cheapest first:

  LEXICAL   token overlap on the harness's own `terms()` -- identifiers first,
            which is the right signal in a codebase. Free, no I/O, no model.

  EMBEDDING the Qwen3-VL embedder on :8099, cosine over Matryoshka-truncated
            vectors. Optional: if the server is down, recall degrades to
            lexical rather than failing.

Lexical runs FIRST and takes a generous shortlist; embedding only re-ranks that
shortlist. That keeps the expensive path proportional to what is about to be
shown, not to the size of the store -- and embedding a whole store would cost
~170 ms per document on this box, which is why it is not the primary scorer.

The embedding scorer is deliberately opt-in (`NEOH_RECALL_EMBED`). Default is
lexical-only, because an unreachable :8099 must not silently make every turn
slower while pretending to be smarter.
"""

from __future__ import annotations

import collections
import json
import math
import os
import time
import urllib.request

# reuse the harness's own term extraction: identifiers first, stopwords out.
# A second, subtly different tokenizer would make scores incomparable with the
# searchers that already use this one.
from .compress import compact  # deterministic, model-free
from .searcher import STOP, WORD_RE

MAX_ENTRIES = int(os.environ.get("NEOH_LTM_MAX_ENTRIES", "512"))
SHORTLIST = int(os.environ.get("NEOH_LTM_SHORTLIST", "8"))     # lexical shortlist
DEFAULT_K = int(os.environ.get("NEOH_RECALL_K", "3"))
EMBED_BASE = os.environ.get("NEOH_RECALL_EMBED_URL", "http://127.0.0.1:8099")
EMBED_MODEL = os.environ.get("NEOH_RECALL_EMBED_MODEL", "qwen3-vl-embedding-2b")
# Matryoshka truncation: measured to hold recall at 64 dims on a 12-doc probe.
EMBED_DIMS = int(os.environ.get("NEOH_RECALL_DIMS", "256"))
EMBED_ENABLED = os.environ.get("NEOH_RECALL_EMBED", "0") == "1"
# Max NEW entries one recall() may embed when the lexical shortlist comes back
# empty. Measured: the embedder is ~0.5 s/call on this box, and a store can hold
# MAX_ENTRIES entries, so an unbounded scan would put minutes inside one turn.
# The scan is incremental on purpose -- each recall does a bounded slice, so a
# large store warms up over several turns instead of stalling the first one.
EMBED_SCAN_BUDGET = int(os.environ.get("NEOH_RECALL_EMBED_BUDGET", "48"))
# Per-entry cap inside render(). A retired span opens with gate markers and the
# content can sit past a head-only cut, so this keeps head AND tail.
ENTRY_CHARS = int(os.environ.get("NEOH_RECALL_ENTRY_CHARS", "400"))


# Conservative suffix folding. NOT a full stemmer -- this exists because a
# lexical recall that scores "deploy" against "deployment" as zero overlap
# silently returns a miss, and a miss looks identical to "nothing was ever
# stored". That failure is invisible in tests that happen to use the same
# wording in the query as in the entry.
#
# Only endings that merge distinct words are stripped: removing too much
# ("ment" from "moment", "ent" from "different") creates false matches.
_SUFFIXES = (
    ("ations", "ate"), ("ation", "ate"), ("ingly", ""), ("edly", ""),
    ("ing", ""), ("ies", "y"), ("ied", "y"), ("ed", ""), ("es", ""),
    ("s", ""), ("ment", ""),
)
# "ers"/"er" are deliberately NOT folded: "server" -> "serv" and "process" ->
# "proces" change the word without merging a real inflection pair, and in a
# codebase `server` and `service` are DIFFERENT identifiers.
#
# The single-letter rules above are guarded so they only fire when the result
# still looks like a word: "process" and "access" end in 's'/'ss' but are not
# plurals, and folding them to "proces"/"acces" merges them with nothing while
# destroying the exact identifier a codebase search depends on.


def _stem(w: str) -> str:
    """Fold a term to a crude stem. Applied to BOTH sides, so it is a matching
    aid, not an index -- the raw term is still kept for display."""
    for suf, rep in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            stem = w[: -len(suf)] + rep
            # Guard the "-s"/"-es" rules: a trailing 'ss' is almost never a
            # plural ("process", "access"), and stripping it produces a stem
            # that matches nothing while corrupting the identifier.
            if suf in ("s", "es") and stem.endswith("s"):
                return w
            return stem
    return w


# Question words carry no retrieval signal but are >= 4 chars, so they survive
# the length filter. Without these, a query like "why did the deploy fail"
# spends 4 of its 5 slots on interrogatives.
_QUESTION = frozenset("""
why how what when where which who whom whose whether
does did doing done can could should would will shall may might must
""".split())


def _terms(text: str, limit: int = 12) -> list[str]:
    """Identifiers first, then long words -- the harness's rule, not a new one,
    plus crude stemming so a query and an entry written in different inflections
    of the same word still meet."""
    ids, words = [], []
    for w in WORD_RE.findall(text or ""):
        lw = w.lower()
        if lw in STOP or lw in _QUESTION:
            continue
        if "_" in w or (w[:1].isupper() and w[1:].lower() != w[1:]):
            ids.append(lw)
        elif len(lw) >= 4:
            words.append(lw)
    out, seen = [], set()
    for w in ids + words:
        key = _stem(w)
        if key not in seen:
            seen.add(key)
            out.append(key)
        if len(out) >= limit:
            break
    return out


def _embed(text: str, dims: int = EMBED_DIMS) -> list[float] | None:
    """Qwen3-VL embedding, truncated to `dims` and renormalised (MRL).

    Returns None if the server is unreachable -- the caller must degrade to
    lexical, never raise. ~130-190 ms per call on this box, CPU-bound, which
    is exactly why it only ever re-ranks a shortlist.
    """
    instr = "Retrieve text relevant to the user's query."
    prompt = (f"<|im_start|>system\n{instr}<|im_end|>\n"
              f"<|im_start|>user\n{text[:4000]}<|im_end|>\n<|im_start|>assistant\n")
    body = json.dumps({"model": EMBED_MODEL, "input": prompt}).encode()
    req = urllib.request.Request(
        EMBED_BASE.rstrip("/") + "/v1/embeddings", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as f:
            v = json.loads(f.read())["data"][0]["embedding"]
    except Exception:                                     # noqa: BLE001
        return None
    v = v[:dims]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else None


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class LTMStore:
    """Bounded store of retired spans, with lexical-then-embedding recall.

    Bounded because it runs in-process with no external store: an unbounded
    dict in a long-horizon agent is a slow leak that shows up as rising prompt
    size. Eviction is strictly LRU by last-hit, so a span that keeps proving
    useful is never the one dropped.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES,
                 embed: bool = EMBED_ENABLED, dims: int = EMBED_DIMS):
        self.max_entries = max(1, max_entries)
        self.embed = embed
        self.dims = dims
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0
        self._embed_calls = 0
        self._embed_failed = 0
        self.fallbacks = 0          # recalls the embedder rescued after lexical came back empty
        self._df: collections.Counter = collections.Counter()   # document freq
        self._n_entries = 0

    def _index(self, entry: dict, sign: int = 1) -> None:
        """Maintain document frequency. Rebuilt on eviction rather than
        decremented, because an evicted entry's terms may appear in others and
        a wrong df silently corrupts every later ranking."""
        self._n_entries += sign
        for t in entry["terms"]:
            self._df[t] += sign
            if self._df[t] <= 0:
                del self._df[t]

    # -- write ------------------------------------------------------------
    def retire(self, text: str, turn: int = -1, meta: dict | None = None) -> dict:
        entry = {
            "id": len(self._entries),
            "text": text,
            "terms": frozenset(_terms(text)),
            "turn": turn,
            "meta": meta or {},
            "at": time.time(),
            "last_hit": 0.0,
            "vec": None,
            # Lossy side-channel for the H_t block. The ORIGINAL above stays the
            # copy recall() searches, so anything compact() drops is still
            # recoverable. Compressing the only copy would make recall unable to
            # find what compaction threw away (docs/ngram-gate-plan.md §11).
            "index": compact(text),
        }
        self._entries.append(entry)
        self._index(entry, +1)

        # Dedup on the TERM SET, not the text.
        #
        # A long run restates the goal on nearly every turn ("Goal: ... (attempt
        # n)"), and those entries are near-identical: measured jaccard 1.000
        # between them, against 0.182 between a goal line and the note that
        # actually answers the question. Left in, they crowd the top-k -- the
        # goal IS a superset match for any query derived from it, so it always
        # ranks first and occupies every slot.
        #
        # The newest wins (the id is kept so callers see a monotonic counter),
        # and the stale duplicate is removed rather than kept as a tiebreak.
        prior = [e for e in self._entries[:-1] if e["terms"] == entry["terms"]]
        if prior:
            stale = prior[-1]
            self._entries.remove(stale)
            self._index(stale, -1)

        if len(self._entries) > self.max_entries:
            # LRU by last_hit; never-evicted entries have last_hit == 0 and are
            # the first to go, which is the right bias for a bounded cache.
            self._entries.sort(key=lambda e: e["last_hit"])
            dropped = self._entries.pop(0)
            self._index(dropped, -1)
            self._reindex(dropped["id"])
        return entry

    def _embed_pending(self, limit: int) -> int:
        """Embed up to `limit` entries that have no vector yet. Returns how many.

        Lazy on purpose. :8099 costs ~0.5 s per call and a store holds up to
        MAX_ENTRIES spans, so recalling must never embed the whole store inside
        one turn. A slice per recall warms a large store over several turns.

        Stops entirely on the first unreachable server: hammering a dead
        endpoint once per entry would turn a missing embedder into the slowest
        thing in the turn, and the caller's fallback is lexical, which is free.
        """
        if not self.embed or limit <= 0:
            return 0
        n = 0
        for e in self._entries:
            if e["vec"] is not None:
                continue
            if n >= limit:
                break
            v = _embed(e["text"], self.dims)
            n += 1
            if v is None:
                self._embed_failed += 1
                return n
            e["vec"] = v
        return n

    def _reindex(self, dropped_id: int) -> None:
        for e in self._entries:
            if e["id"] > dropped_id:
                e["id"] -= 1

    # -- read -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def chars(self) -> int:
        return sum(len(e["text"]) for e in self._entries)

    def recall(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        """Top-k entries for `query`.

        Lexical shortlist first, then optional embedding re-rank of just that
        shortlist. Returns [] rather than raising when nothing is stored --
        a recall miss is a normal outcome, not an error.
        """
        if not self._entries or not query:
            return []
        qt = set(_terms(query, limit=16))
        if not qt:
            return []

        scored = []
        for e in self._entries:
            if not e["terms"]:
                continue
            inter = qt & e["terms"]
            if not inter:
                continue
            # IDF weighting, NOT term count.
            #
            # Counting shared terms scores the Goal line above the fact that
            # actually answers the question, because the goal is echoed nearly
            # verbatim by the query AND appears in every window -- so it is
            # guaranteed to look maximally relevant. That is exactly backwards.
            #
            # A term present in every entry carries no information, so it must
            # count for less.
            #
            # The normaliser is the ENTRY's own matched-term count, not the
            # query's: an entry that matches 4 of 4 query terms is not "twice
            # as relevant" as one matching 2 of 4, it is the only one that
            # covers the query at all. Dividing by a per-QUERY constant (the
            # first attempt) leaves every entry with the same denominator and
            # so cannot discriminate between them.
            hits_n = len(inter)
            if not hits_n:
                continue
            score = sum(math.log(1.0 + self._n_entries / (1.0 + self._df[t]))
                        for t in inter) / hits_n
            # Coverage bonus: an entry matching more DISTINCT query terms is
            # more likely to contain the answer, and this is what lifts the
            # fact above the goal line once the shared terms are discounted.
            score *= (1.0 + 0.15 * (hits_n - 1))
            scored.append((score, e))
        scored.sort(key=lambda kv: -kv[0])
        shortlist = [e for _, e in scored[:SHORTLIST]]

        # Lexical found nothing: this is where the embedder earns its place.
        #
        # Measured on 20 retired spans / 14 labelled queries (docs/ngram-gate-plan.md §8):
        # a lexical-empty query returned [] and never reached the embedder at all --
        # 5 of 14 -- while the embedder alone got those right. Re-ranking the lexical
        # shortlist instead (the previous behaviour) changed NOTHING: 8/14 either
        # way. A term matcher cannot see that "where do we keep passwords" wants the
        # span that says "Secrets live in Vault"; the embedder can, and it was being
        # switched off precisely where it wins.
        #
        # Ordering stays lexical-first on purpose: where lexical already ranks well
        # its order is not disturbed, so the fallback can only add candidates that
        # were previously invisible -- it cannot demote a good lexical hit.
        if not shortlist and self.embed:
            failures_before = self._embed_failed
            self._embed_pending(EMBED_SCAN_BUDGET)
            # _embed_pending may have just discovered the server is gone. Do not
            # spend a second timeout proving it; the lexical path below is free.
            qv = _embed(query, self.dims) if self._embed_failed == failures_before else None
            if qv is not None:
                self._embed_calls += 1
                rescored = [(_cos(qv, e["vec"]), e) for e in self._entries if e["vec"] is not None]
                rescored.sort(key=lambda kv: -kv[0])
                shortlist = [e for _, e in rescored[:SHORTLIST]]
                self.fallbacks += 1

        if not shortlist:
            self.misses += 1
            return []

        if self.embed:
            qv = _embed(query, self.dims)
            if qv is None:
                self._embed_failed += 1
            else:
                self._embed_calls += 1
                rescored = []
                for e in shortlist:
                    if e["vec"] is None:
                        e["vec"] = _embed(e["text"], self.dims)
                    if e["vec"] is not None:
                        rescored.append((_cos(qv, e["vec"]), e))
                # re-ranks ONLY the shortlist; entries the embedder could not
                # score fall back to their lexical rank rather than vanishing
                if rescored:
                    rescored.sort(key=lambda kv: -kv[0])
                    shortlist = [e for _, e in rescored]

        out = shortlist[:k]
        now = time.time()
        for e in out:
            e["last_hit"] = now
        self.hits += len(out)
        return out

    def render(self, query: str, k: int = DEFAULT_K,
               budget_chars: int = 1200) -> str:
        """A compact block for H_t.

        Sized in CHARACTERS, not entries: k entries of unknown length would
        blow the budget, and blowing the budget is the failure this whole gate
        exists to prevent. States how much was dropped (law hand_cut_says).
        """
        hits = self.recall(query, k)
        if not hits:
            return ""
        lines, used, dropped = [], 0, 0
        for e in hits:
            # The compacted index first; the original only if compaction did not pay.
            body = (e.get("index") or e["text"]).strip().replace("\n", " ")
            if len(body) > ENTRY_CHARS:
                # Keep the TAIL as well as the head. A retired span opens with
                # gate markers ("[gate: retired 3 turn(s)...] [user] tool error
                # ...") and the actual content can sit past a head-only cut, so
                # a 300-char prefix showed the marker and dropped the fact.
                head = ENTRY_CHARS * 2 // 3
                body = (body[:head] + f" ...[{len(body) - ENTRY_CHARS} chars elided]"
                        + body[-(ENTRY_CHARS - head):])
            line = f"- (turn {e['turn']}) {body}"
            if used + len(line) > budget_chars:
                dropped += 1
                continue
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        head = f"### Long-term memory (recall for: {query[:60]})"
        if dropped:
            head += f" [{dropped} further match(es) not shown: over budget]"
        return head + "\n" + "\n".join(lines)

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "chars": self.chars,
            "hits": self.hits,
            "misses": self.misses,
            "embed_enabled": self.embed,
            "embed_calls": self._embed_calls,
            "embed_failures": self._embed_failed,
            "embed_fallbacks": self.fallbacks,
            "compacted_chars": sum(len(e.get("index") or "") for e in self._entries),
            "stored_chars": self.chars,
            "max_entries": self.max_entries,
        }
