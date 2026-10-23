"""Backplane law invariants for the neoharness fan-out.

Each test is a direct transcription of a law from
AgentFrameworks/AgentFrameworkTODOS -> Backplane/LAWS.bend. The laws are the
contract ("Humans own it. An agent may add laws, but never weaken or delete one
without the user's explicit consent"), so this file pins the behaviour neoharness
owes them BEFORE the behaviour is changed.

Sources:
  law handoff_goes_first      - "it goes first, ahead of the user's words, which are never cut"
  law handoff_never_runs_alone- "context waiting alone never starts a turn: it goes with a message"
  law hand_no_holes           - "once one entry is left out, every older one is too"
  law hand_newest_fit         - budgeted selection keeps the newest that fit
  law hand_cut_says           - "a long item is cut and says by how much"
  law cron_due_fires_once     - "fires a routine once for any number of missed times,
                                 and never twice at the same moment"
"""

from __future__ import annotations

import threading

import pytest

from neoharness.agent import Agent


# ---------------------------------------------------------------- handoff queue

def test_handoff_goes_first():
    """law handoff_goes_first:
       M.Queue.texts([QItem{"handoff:t1","ctx"}, QItem{"m1","hi"}]) == "ctx\\n\\nhi"
       The handoff leads; the user's own words are never shortened to make room."""
    from neoharness.handoff import queue_texts
    items = [("handoff:t1", "ctx"), ("m1", "hi")]
    assert queue_texts(items) == "ctx\n\nhi"


def test_handoff_never_runs_alone():
    """law handoff_never_runs_alone: Queue.real([handoff]) == False
       law handoff_with_message_runs:  Queue.real([handoff, message]) == True
       Context alone never starts a turn; it rides along with a real message."""
    from neoharness.handoff import queue_real
    assert queue_real([("handoff:t1", "ctx")]) is False
    assert queue_real([("handoff:t1", "ctx"), ("m1", "hi")]) is True


# ---------------------------------------------------------------- Hand: budgeted selection

def test_hand_cut_says():
    """law hand_cut_says: HD.Hand.cut("abcdef", 2n) == "ab ...[4 more characters]"
       A cut item says by how much it was cut - it never silently truncates."""
    from neoharness.handoff import cut
    assert cut("abcdef", 2) == "ab ...[4 more characters]"


def test_hand_newest_fit():
    """law hand_newest_fit: with budget 17 over three items (a=1,b=2,c=3, oldest
       first) the result is Sel{left:1, kept:["User: b","User: c"], omitted:1}.
       The newest items that fit are kept."""
    from neoharness.handoff import select
    sel = select(["User: a", "User: b", "User: c"], budget=17)
    assert sel.kept == ["User: b", "User: c"]
    assert sel.left == 1        # the law's exact Sel{1n, [...], True{}, 1n}
    assert sel.omitted == 1
    assert sel.truncated is True


def test_hand_no_holes():
    """law hand_no_holes: once one entry is left out, every older one is too.
       The kept history is a SUFFIX of the input - never a selection with gaps.
       A summary that keeps items 1 and 3 but drops 2 is a hole."""
    from neoharness.handoff import select
    items = [f"User: item{i}" for i in range(6)]
    for budget in (10, 25, 40, 60, 90, 120):
        sel = select(items, budget=budget)
        kept_n = [int(k.rsplit("item", 1)[1]) for k in sel.kept]
        # kept items must be a contiguous suffix, and omitted == everything older
        assert kept_n == list(range(6 - len(kept_n), 6)), f"hole at budget={budget}"
        assert sel.omitted == 6 - len(kept_n)


# ---------------------------------------------------------------- the turn gate

class CountingStub:
    """A stub LLM that counts calls, so 'did this expand again?' is observable."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()
        self._n = 0

    def chat(self, messages, **kw):
        with self._lock:
            self.calls += 1
            self._n += 1
            n = self._n
        # 1 = the four sub-goals; anything else = a bare reply
        return "1. a\n2. b\n3. c\n4. d" if n == 1 else "<done>ok</done>"


def test_gate_fires_once_per_boundary():
    """law cron_due_fires_once: "fires a routine once for any number of missed
       times, and never twice at the same moment".

       Two delegate calls inside the SAME turn are the same moment: the second
       must not run another 1:4:1 expansion."""
    llm = CountingStub()
    agent = Agent(llm, fanout=4, turn_len=4, use_bpe=False, max_steps=8)
    delegate = agent.tools.tools["delegate"]

    agent._turn = 0                      # phase 0 - the seam is open
    delegate.run(goal="first", context="")
    after_first = llm.calls

    delegate.run(goal="second", context="")   # same turn == same moment
    after_second = llm.calls

    assert after_second == after_first, (
        f"the gate expanded twice in turn 0: {after_first} -> {after_second} calls")


def test_gate_still_closed_off_boundary():
    """The gate that fires once must still be a gate: inside a turn it refuses."""
    llm = CountingStub()
    agent = Agent(llm, fanout=4, turn_len=4, use_bpe=False, max_steps=8)
    delegate = agent.tools.tools["delegate"]

    agent._turn = 2                      # phase 2/4 - closed
    out = delegate.run(goal="anything", context="")
    assert "gate closed" in out
    assert llm.calls == 0, "a closed gate must not call the model at all"


# ------------------------------------------- the consolidation must not summarise

def test_consolidation_is_verbatim_not_a_summary():
    """handoff.bend: the handoff is "a budgeted selection of the source thread's
    own messages and tool lines, NEVER AN AGENT-WRITTEN SUMMARY".

    The consolidation must carry the workers' own words through unchanged, and
    name anything it drops. It must NOT ask a model to merge them: a summary
    loses facts with no trace, which is precisely what the contract forbids."""
    class L:
        def chat(self, m, **k):
            raise AssertionError("_consolidate called the model -> that is an agent-written summary")
        usage = {}
    a = Agent(L(), fanout=4, turn_len=4, use_bpe=False, max_steps=8)
    reports = [f"[w{i}] MARKER{i} verbatim text {i}" for i in range(1, 5)]
    out = a._consolidate("goal", reports)
    for i in range(1, 5):
        assert f"MARKER{i} verbatim text {i}" in out, f"worker {i} was not carried verbatim"
    assert "left out" not in out          # all four fit, so nothing is dropped
    assert out.startswith("[1:4:1 expansion]")   # handoff_goes_first: the banner leads


def test_consolidation_names_what_it_drops():
    """hand_no_holes + hand_cut_says: over budget, the oldest go first and the
    count is stated; the kept set is a suffix, not a spliced selection."""
    class L:
        def chat(self, m, **k):
            raise AssertionError("no model call allowed in consolidation")
        usage = {}
    a = Agent(L(), fanout=4, turn_len=4, use_bpe=False, max_steps=8)
    # each worker is capped at CAP=4000 by hand_cut_says, so four of them are
    # 4*4026+4 = 16108 chars against a 16000 budget: exactly one falls off the
    # front and three are carried (a suffix, not a splice).
    big = [f"[w{i}] " + "X" * 8000 + f" END{i}" for i in range(1, 5)]
    out = a._consolidate("goal", big)
    assert "left out" in out, "dropped workers were not named"
    assert "3 carried verbatim" in out, "the count of carried reports is wrong"
    # suffix: the earliest worker is the one dropped, the last three survive
    assert "[worker 1]" not in out
    for i in (2, 3, 4):
        assert f"[worker {i}]" in out
    # every carried worker says how much of itself was cut (hand_cut_says)
    assert "more characters]" in out
