"""Tests for overlapping goals: the 1:4 duty-server setup.

Two things have to hold once more than one goal is in flight:

1. serve.py must not serialize on a mutex -- a continuous-batching duty
   server (llama-server -np 4) can only fill its slots if more than one
   request is resident at a time.
2. The shared BPEState is mutated by every goal, so its stores must be
   serialized; otherwise evict/append interleavings corrupt the banks.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from neoharness.bpe import BPEState
from neoharness.serve import make_server


class _SlowLLM:
    """Stalls in chat() so overlap is observable, and counts peak concurrency."""

    def __init__(self, hold: float = 0.25):
        self.hold = hold
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def chat(self, messages, **kw):
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self.hold)
            return "<done>ok</done>"
        finally:
            with self._lock:
                self.live -= 1


def _post(port, payload: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _fire(port, goal: str):
    return _post(port, {"messages": [{"role": "user", "content": goal}]})


@pytest.fixture()
def concurrent_server(tmp_path):
    llm = _SlowLLM()
    srv = make_server("127.0.0.1", 0, llm, tmp_path, max_concurrent_goals=4)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, llm
    srv.shutdown()
    srv.server_close()


def test_goals_overlap_when_allowed(concurrent_server):
    """Four goals in flight at once -> the duty server sees four busy slots."""
    srv, llm = concurrent_server
    port = srv.server_address[1]
    t0 = time.time()
    threads = [threading.Thread(target=_fire, args=(port, f"goal {i}"))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert all(not t.is_alive() for t in threads)
    assert llm.peak == 4, f"only {llm.peak} goals overlapped"


def test_concurrency_one_stays_serial(tmp_path):
    """The default (1) must keep the old serial behaviour."""
    llm = _SlowLLM(hold=0.05)
    srv = make_server("127.0.0.1", 0, llm, tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        threads = [threading.Thread(target=_fire, args=(port, f"g{i}"))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert llm.peak == 1, "default must not overlap goals"
    finally:
        srv.shutdown()
        srv.server_close()


def test_bpe_state_survives_concurrent_mutation(tmp_path):
    """Concurrent writers must not lose entries or trip eviction invariants."""
    st = BPEState()
    errors: list[BaseException] = []

    def worker(n: int):
        try:
            for i in range(40):
                st.belief.update(f"obj{n}-{i}", state="here")
                st.progress.commit(f"sub {n}-{i}", "done")
                st.experience.add("general", f"t{n}-{i}", "body")
                st.experience.recall(f"t{n}-{i}")
                st.render()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"worker raised: {errors!r}"
    assert len(st.belief.facts) <= st.belief.max_facts
    assert len(st.progress.entries) <= st.progress.max_subgoals
    assert len(st.experience.skills) <= st.experience.max_skills
    assert st.progress.entries, "concurrent commits lost everything"
    # every surviving entry is well-formed (no torn dicts)
    assert all("subgoal" in e and "status" in e for e in st.progress.entries)
    assert all("skill_id" in e for e in st.experience.skills.values())


def test_concurrent_save_is_atomic(tmp_path):
    """Interleaved saves must leave a parseable whole file, never a torn one."""
    st = BPEState()
    target = tmp_path / "bpe_state.json"
    errors: list[BaseException] = []

    def worker(n: int):
        try:
            for i in range(30):
                st.experience.add("general", f"t{n}-{i}", "body")
                st.save(target)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"save raised: {errors!r}"
    data = json.loads(target.read_text())  # parses => os.replace kept it whole
    assert set(data) == {"belief", "progress", "experience"}