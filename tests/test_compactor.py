"""Tests for compactor.py — budget fit, elision note, latent transfer, keep-turns."""

from __future__ import annotations

from neoharness.compactor import ContextCompactor


class FakeBuffer:
    def __init__(self):
        self.stores = []

    def store_axis(self, axis, text, persist=False, mode="merge"):
        self.stores.append({"axis": axis, "text": text, "persist": persist})
        return {"ok": True}


def _msgs(n, size=1000):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"turn{i} " + "x" * size})
    return msgs


def test_under_budget_untouched():
    c = ContextCompactor(fit_chars=100000)
    msgs = _msgs(4)
    assert c.fit(msgs) == msgs


def test_over_budget_elides_oldest_keeps_tail():
    c = ContextCompactor(fit_chars=4600, keep_turns=3)
    msgs = _msgs(8)
    out = c.fit(msgs)
    assert out[0]["role"] == "system"
    # tail preserved verbatim
    assert out[-3:] == msgs[-3:]
    notes = [m for m in out if "elided" in m["content"]]
    assert notes and "transferred" not in notes[0]["content"]  # no buffer
    assert sum(len(m["content"]) for m in out) <= 4600 + 200  # note overhead


def test_latent_transfer_before_elision():
    buf = FakeBuffer()
    c = ContextCompactor(buffer=buf, fit_chars=4600, keep_turns=3)
    out = c.fit(_msgs(8))
    assert buf.stores, "elided span must be offered to the buffer"
    store = buf.stores[0]
    assert store["axis"] == "x" and store["persist"] is False
    assert "turn0" in store["text"]  # oldest content was transferred
    assert any("transferred to n-gram buffer" in m["content"] for m in out)
    assert c.transfers and c.transfers[0]["stored"] is True


def test_failed_transfer_still_elides():
    class DeadBuffer:
        def store_axis(self, axis, text, persist=False, mode="merge"):
            return None

    c = ContextCompactor(buffer=DeadBuffer(), fit_chars=4600, keep_turns=3)
    out = c.fit(_msgs(8))
    assert any("elided" in m["content"] for m in out)
    assert c.transfers[0]["stored"] is False


def test_keep_turns_floor_respected():
    c = ContextCompactor(fit_chars=10, keep_turns=4)
    msgs = _msgs(4)  # system + 4 turns, budget impossible
    out = c.fit(msgs)
    assert out == msgs  # nothing elidable; gives up instead of eating tail
