"""Tests for ltm.py -- the readable side of the sliding gate.

The bug these exist to prevent: a lexical recall that scores "deploy" against
"deployment" as zero overlap returns a MISS, and a miss is indistinguishable
from "nothing was ever stored". That is invisible in any test where the query
and the entry happen to use the same wording.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoharness.ltm import LTMStore, _stem, _terms  # noqa: E402
import neoharness.ltm as L  # noqa: E402  (module ref: the fallback tests monkeypatch L._embed)
import pytest  # noqa: E402


def _store(**kw):
    # `embed` defaults to False here because most tests in this module are about
    # lexical recall and must not depend on :8099 being up; an explicit
    # embed=True in kw still wins, for the fallback tests below.
    return LTMStore(**{"embed": False, **kw})


# ---- the recall bug this module was written to fix -----------------------

def test_inflection_still_matches():
    """'deploy' must reach an entry that says 'deployment'. This is the exact
    miss that made recall look empty when it was not."""
    s = _store()
    s.retire("the deployment failed because the database migration rolled back")
    hits = s.recall("why did the deploy fail")
    assert hits, "stemmed query failed to reach a stored entry"
    assert "deployment" in hits[0]["text"]


def test_stemmer_merges_real_inflections():
    for a, b in (("deployment", "deploy"), ("failed", "fail"),
                 ("migrations", "migration"), ("observations", "observation"),
                 ("rolled", "roll")):
        assert _stem(a) == _stem(b), f"{a} and {b} should meet: {_stem(a)} != {_stem(b)}"


def test_stemmer_does_not_over_merge():
    """Over-stemming creates false matches, which is worse than a miss: it puts
    the WRONG entry in front of the model."""
    for w in ("server", "service", "moment", "different", "process", "access"):
        assert _stem(w) == w, f"{w} was over-stemmed to {_stem(w)}"


def test_question_words_are_not_search_terms():
    q = _terms("why did the deploy fail")
    assert "why" not in q and "did" not in q
    assert q == ["deploy", "fail"], f"query terms not signal-only: {q}"


# ---- recall behaviour -----------------------------------------------------

def test_recall_returns_the_relevant_entry():
    s = _store()
    s.retire("the deployment failed because the database migration rolled back", turn=1)
    s.retire("pasta needs fresh basil and olive oil", turn=2)
    s.retire("def alpha_main() returns 'alpha'", turn=3)
    for q, expect in (("why did the deploy fail", "deployment"),
                      ("what is the pasta recipe", "pasta"),
                      ("alpha_main", "alpha_main")):
        hits = s.recall(q, k=1)
        assert hits, f"miss on {q!r}"
        assert expect in hits[0]["text"], f"{q!r} -> {hits[0]['text'][:60]!r}"


def test_recall_miss_is_empty_not_an_error():
    s = _store()
    s.retire("pasta needs basil")
    assert s.recall("unrelated quantum chromodynamics") == []
    assert s.recall("") == []
    assert s.recall("anything") == []          # empty store


def test_identifier_wins_over_prose():
    """The harness rule: in a codebase, `deploy_gate` beats a long sentence."""
    s = _store()
    s.retire("deploy_gate gates the turn loop and refuses inside the cycle")
    s.retire("the quick brown fox jumps over the lazy dog repeatedly today")
    hits = s.recall("deploy_gate", k=1)
    assert hits and "deploy_gate" in hits[0]["text"]


# ---- bounding -------------------------------------------------------------

def test_render_is_char_bounded_and_says_what_it_dropped():
    """law hand_cut_says: a cut item says by how much. And the budget is in
    CHARACTERS because entries have unknown length.

    The entries must be DISTINCT: retire() dedups on the term set, and twenty
    copies of "record N detail detail..." share the term set {record, detail},
    so they collapse to one and there is nothing left to drop. That version of
    this test asserted a marker that could never appear.
    """
    s = _store()
    for i in range(20):
        s.retire(f"record {i} topic_topic{i} detail{i} item{i} part{i} "
                 + "filler " * 30, turn=i)
    assert len(s) > 1, "test setup: everything deduped away"
    out = s.render("record", k=20, budget_chars=300)
    assert out, "render produced nothing"
    assert len(out) <= 300 + 160, f"render blew its budget: {len(out)}"
    assert "not shown" in out, "a dropped match must be declared"


def test_duplicate_spans_collapse():
    """A long run restates the goal every turn. Those entries are near-
    identical (measured jaccard 1.000) and, being superset matches for any
    query derived from the goal, would otherwise occupy every top-k slot and
    crowd out the one entry that answers the question."""
    s = _store()
    q = "report the vault code for obj_delta as only the digits"
    for i in range(6):
        s.retire(f"[user] Goal: {q} (attempt {i})", turn=i)
    s.retire('[assistant] <tool>note {"text": "vault code is 7391"}</tool>', turn=7)
    assert len(s) == 2, f"duplicates did not collapse: {len(s)} entries"
    # and the distinct entry is now reachable at all
    hits = s.recall(q, k=3)
    assert any("note" in e["text"] for e in hits), "the distinct entry is unreachable"


def test_store_is_bounded_and_evicts_lru():
    """An unbounded dict in a long-horizon agent is a slow leak that shows up
    as rising prompt size."""
    s = _store(max_entries=8)
    for i in range(30):
        s.retire(f"entry number {i} about topic_{i}")
    assert len(s) == 8, f"store not bounded: {len(s)}"

    # an entry that keeps being recalled must survive
    s.retire("the vault code for obj_delta is 7391", turn=99)
    for _ in range(5):
        s.recall("vault code obj_delta", k=1)
    hits = s.recall("vault code obj_delta", k=1)
    assert hits and "7391" in hits[0]["text"], "hot entry was evicted"


# ---- integration with the gate -------------------------------------------

def test_gate_retirement_lands_in_a_readable_store():
    """The end-to-end property: what the gate retires must be recallable, or
    the cycle is rotating blind."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neoharness.sliding_gate import SlidingGate

    gate = SlidingGate(slot_chars=1500, stm_slots=3, turn_len=8, period=16)
    store = _store()
    msgs = [{"role": "system", "content": "SYS"}]
    # plant a fact early; it must survive the window being compacted past it
    msgs.append({"role": "user", "content": "Remember: the vault code for "
                                           "obj_delta is 7391 and the owner is team 4."})
    for t in range(14):
        msgs = msgs + [
            {"role": "user", "content": f"Observation {t}: step {t} " + "filler " * 300},
            {"role": "assistant", "content": "ok"},
        ]
        out = gate.apply(msgs, t)
        # whatever the gate dropped, hand it to the store
        for r in gate.retired[-1:]:
            if r.get("text"):
                store.retire(r["text"], turn=t)
        msgs = out
    # the plant is no longer in the window ...
    assert not any("7391" in (m.get("content") or "") for m in msgs), \
        "test setup: the fact is still live, nothing was retired"
    # ... but it is recoverable
    hits = store.recall("vault code obj_delta", k=1)
    assert hits and "7391" in hits[0]["text"], "retired fact is not recallable"


def test_stats_expose_embed_state():
    """An unreachable embedder must be visible, not silent."""
    s = _store()
    s.retire("something")
    s.recall("something")
    st = s.stats()
    assert st["embed_enabled"] is False
    assert st["embed_calls"] == 0
    assert set(st) >= {"entries", "chars", "hits", "misses", "max_entries"}


# ---- the embedding fallback (union path) -----------------------------------
# Measured in docs/ngram-gate-plan.md §8: with a LEXICAL-EMPTY query the old code
# returned [] and never called the embedder, so the span it should have found was
# invisible. 5 of 14 labelled queries hit this. Printing a number is not a test;
# these pin the property.

def test_lexical_empty_query_is_rescued_by_the_embedder(monkeypatch):
    """The whole point of the union: a query sharing NO terms with the answer
    must still reach it. Lexical recall here is genuinely empty."""
    st = _store(embed=True)
    st.retire("Secrets live in Vault under path platform/<env>/<service>.")
    st.retire("Ran the test suite: 412 passed, 3 skipped, 0 failed.")

    calls = []

    def fake_embed(text, dims=256):
        calls.append(text)
        if text.startswith("system"):
            return [1.0] + [0.0] * (dims - 1)          # the query
        return [0.0] + [1.0] * (dims - 1) if "Vault" in text else [0.5] * dims

    monkeypatch.setattr(L, "_embed", fake_embed)
    hits = st.recall("where do we keep passwords and keys?", k=1)
    assert hits, "a lexical-empty query returned nothing; the embedder never ran"
    assert "Vault" in hits[0]["text"]
    assert st.fallbacks == 1, "the rescue must be counted, not silent"


def test_fallback_does_not_fire_when_lexical_already_answers():
    """The fallback is a FALLBACK: a store-wide embedding scan must not run on a
    turn that lexical already answers.

    Note the embedder is still consulted for the pre-existing shortlist RE-RANK
    (bounded by SHORTLIST entries, which is a different and deliberate path).
    What must never happen is the fallback's scan over entries lexical did not
    pick -- that is the expensive one. `fallbacks` counts exactly that.
    """
    st = _store(embed=True)
    for i in range(40):                       # a store big enough to scan
        st.retire(f"unrelated entry {i} zebra")
    st.retire("The p99 latency budget is 250ms end to end.")
    hits = st.recall("what is the p99 latency budget?", k=1)
    assert hits and "250ms" in hits[0]["text"]
    assert st.fallbacks == 0, "the fallback scanned the store on a lexical hit"
    unscanned = sum(1 for e in st._entries if e["vec"] is None)
    # The only entries that may be embedded here are the lexical shortlist's,
    # which the pre-existing re-rank touches. Anything more would be the scan.
    assert unscanned >= len(st._entries) - L.SHORTLIST, (
        f"a lexical answer embedded {len(st._entries) - unscanned} entries; "
        f"the re-rank may touch at most {L.SHORTLIST}")


def test_one_recall_embeds_at_most_the_budget(monkeypatch):
    """The embedder costs ~0.5 s/call and a store holds up to MAX_ENTRIES spans.
    A recall that embedded the whole store would put minutes inside one turn.

    The spans must have DISTINCT term sets: retire() dedups on the term set
    (see its docstring), so "entry 0".."entry 199" would collapse to one entry
    and this test would pass vacuously. That is exactly what happened the first
    time it was written.
    """
    st = _store(embed=True, max_entries=200)
    for i in range(200):
        st.retire(f"observation{i} detail{i} marker{i}")
    assert len(st._entries) == 200, "spans collapsed; the budget would not be exercised"

    seen = []

    def counting(text, dims=256):
        seen.append(text)
        return [0.1] * dims

    monkeypatch.setattr(L, "_embed", counting)
    monkeypatch.setattr(L, "EMBED_SCAN_BUDGET", 10, raising=False)
    st.recall("zzqq unrelated xxww vvuu", k=3)
    embedded = sum(1 for e in st._entries if e["vec"] is not None)
    assert embedded <= 10, f"one recall embedded {embedded} entries, budget was 10"


def test_unreachable_embedder_stops_after_one_call(monkeypatch):
    """A dead :8099 must cost ONE failed call, not one per entry.

    The pre-existing behaviour degraded to lexical correctly but kept asking:
    a store of 60 spans meant 60 timeouts inside one turn, which turns a
    missing optional dependency into the slowest thing in the turn.
    """
    st = _store(embed=True)
    for i in range(60):
        st.retire(f"observation{i} detail{i} marker{i}")
    assert len(st._entries) == 60

    calls = []

    def dead(text, dims=256):
        calls.append(text)
        return None

    monkeypatch.setattr(L, "_embed", dead)
    hits = st.recall("zzqq unrelated xxww vvuu", k=3)
    assert len(calls) == 1, f"a dead embedder was called {len(calls)} times in one recall"
    assert st._embed_failed == 1
    assert hits == [], "nothing lexical matched, so nothing should be returned"


def test_unreachable_embedder_degrades_to_lexical_not_an_error(monkeypatch):
    """Precedent: the embedder is opt-in and must never turn a turn into an
    exception. An unreachable :8099 still yields the lexical answer."""
    st = _store(embed=True)
    st.retire("The p99 latency budget is 250ms end to end.")
    monkeypatch.setattr(L, "_embed", lambda *a, **k: None)
    hits = st.recall("what is the p99 latency budget?", k=1)
    assert hits and "250ms" in hits[0]["text"], "a dead embedder cost us a lexical hit"
    assert st._embed_failed >= 1


def test_stats_expose_the_fallback_count():
    st = _store(embed=False)
    assert st.stats()["embed_fallbacks"] == 0
