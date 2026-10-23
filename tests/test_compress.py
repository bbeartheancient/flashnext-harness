"""Deterministic compaction (neoharness/compress.py) and the two-tier store law.

The law that matters: compaction is a LOSSY SIDE-CHANNEL. render() may show a
compressed fact index; recall() must keep searching the ORIGINAL, or compaction
would destroy the very text recall depends on. Printing a number is not a test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoharness.compress import compact, ratio_of, MIN_CHARS  # noqa: E402
from neoharness.ltm import LTMStore  # noqa: E402

SPAN = (
    "[gate: retired 4 turn(s) ago] [user] the checkout flow times out in staging again. "
    "I read checkout-service and the timeout is set to 250ms but the real p99 on that path "
    "is 310ms because we added the tax calculation in March. Raising it to 400ms fixed "
    "staging, but the error budget for the team is 3% and we are now at 3.4% weekly, so I "
    "did not ship it to prod. The tax table is cached in Redis under tax:v3:{region} with "
    "a 15 minute TTL. The March change bypassed the cache on the refund path only. "
    "The payment client retries idempotent calls 3 times with jittered backoff, but it must "
    "never retry on 4xx. There is a known bug where a 409 from the tax service is treated as "
    "retryable; ticket PAY-4471, owner Priya Raman, due 2026-10-16. We do not need a "
    "migration for the redis change, it is a cache key change only. Rollback is to revert "
    "the timeout to 250ms, and the tax cache change is revertible via the TAX_CACHE_ENABLED "
    "flag which is currently off by default in all environments. "
)


# ---- the compressor -------------------------------------------------------

def test_compaction_actually_shrinks():
    out = compact(SPAN)
    assert len(out) < len(SPAN), "compaction returned something no smaller"
    assert ratio_of(SPAN, out) > 1.5, f"only {ratio_of(SPAN, out):.2f}x, expected >1.5x"


def test_compaction_keeps_most_load_bearing_facts():
    """Lossy by design - a fact-index at ~2.5x cannot keep everything.

    What it must do is keep the MAJORITY, and what it drops must still be
    recoverable from the original (the two-tier law below). Asserting 100%
    here would just push the ratio up and give back the compression.
    """
    out = compact(SPAN)
    wanted = ("250ms", "400ms", "3.4%", "tax:v3", "PAY-4471", "Priya Raman",
              "2026-10-16", "TAX_CACHE_ENABLED")
    kept = [t for t in wanted if t in out]
    assert len(kept) >= 6, f"compaction kept only {len(kept)}/{len(wanted)}: " \
                           f"lost {[t for t in wanted if t not in out]}"


def test_short_spans_are_left_alone():
    """Below MIN_CHARS the machinery costs more than it saves."""
    short = "just a short note about the deploy"
    assert compact(short) == short


def test_compaction_is_deterministic():
    """Two calls must agree, or H_t changes for no reason between turns."""
    assert compact(SPAN) == compact(SPAN)


def test_compaction_never_returns_something_longer():
    for s in (SPAN, SPAN * 2, SPAN[:300]):
        assert len(compact(s)) <= len(s)


# ---- the two-tier law -----------------------------------------------------

def test_recall_searches_the_original_not_the_index():
    """THE law. A fact the index dropped must still be findable."""
    st = LTMStore(embed=False)
    st.retire(SPAN)
    hits = st.recall("who owns PAY-4471?")
    assert hits, "recall found nothing in a span it stored"
    assert "Priya Raman" in hits[0]["text"], \
        "recall searched the compressed index instead of the original"


def test_render_puts_the_smaller_index_in_context():
    """render() must show the index, not the original, or compaction is decorative."""
    st = LTMStore(embed=False)
    st.retire(SPAN)
    idx = st._entries[0]["index"]
    assert idx != SPAN and len(idx) < len(SPAN)
    out = st.render("PAY-4471 retry 4xx")
    assert out
    assert "Priya Raman" in out, "render did not use the index for a fact it holds"


def test_a_fact_the_index_dropped_is_still_recallable():
    """The two tiers together: loss in the index is covered by the original.

    400ms is lost by compaction at the default ratio (measured). It must still
    come back through recall, which searches e["text"], never e["index"].
    """
    st = LTMStore(embed=False)
    st.retire(SPAN)
    assert "400ms" not in st._entries[0]["index"], \
        "premise broken: this span's index happens to keep 400ms; pick a tighter ratio"
    assert "400ms" in st._entries[0]["text"], "the original lost it"
    hits = st.recall("timeout 400ms staging")
    assert any("400ms" in h["text"] for h in hits), \
        "recall could not recover a fact the index dropped"


def test_the_original_is_still_stored_in_full():
    st = LTMStore(embed=False)
    st.retire(SPAN)
    assert st._entries[0]["text"] == SPAN, "the stored copy was compacted in place"


def test_stats_shows_both_tiers():
    st = LTMStore(embed=False)
    st.retire(SPAN)
    s = st.stats()
    assert s["stored_chars"] == len(SPAN)
    assert 0 < s["compacted_chars"] < s["stored_chars"]


def test_clauses_come_back_in_reading_order():
    """Score order is fine for SELECTION; it must not leak into the OUTPUT.

    A fact index whose clauses jump around is still deterministic, so the
    determinism test above cannot catch this - it needs its own.
    """
    from neoharness.compress import _clauses
    out = compact(SPAN)
    kept = [c for c in _clauses(out)]
    pos = []
    for c in kept:
        i = SPAN.find(c[:40])
        assert i >= 0, f"output clause is not from the source: {c[:40]!r}"
        pos.append(i)
    assert pos == sorted(pos), "clauses came back in score order, not reading order"
