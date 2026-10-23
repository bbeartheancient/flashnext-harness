"""Tests for sliding_gate.py — the running-process compaction gate.

The two regression tests (`test_boundary_never_retreats`,
`test_window_bounded_under_continuous_growth`) exist because both properties
were violated in earlier versions of this module and neither showed up as a
crash: the gate returned plausible-looking windows that carried 53% MORE than
the threshold policy it was meant to replace. Printing a number is not a test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoharness.sliding_gate import SlidingGate  # noqa: E402


def _msgs(n_turns=10, turn_chars=3000, head_chars=13000):
    msgs = [{"role": "system", "content": "S" * head_chars}]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"### Harness state (H_t)\nturn {t}\n\ngoal {t}"})
        msgs.append({"role": "assistant", "content": "y" * turn_chars})
    return msgs


# ---- invariants -----------------------------------------------------------

def test_system_head_is_never_retired():
    """The stable head is the whole point: if a system message is rotated
    out, the cache prefix dies and every turn re-prefills from scratch."""
    g = SlidingGate(slot_chars=2000, stm_slots=2, turn_len=8)
    msgs = _msgs(12)
    head = "S" * 13000
    for t in range(12):
        msgs = g.apply(msgs, t)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == head, "system head mutated"


def test_apply_does_not_mutate_input():
    """agent.py keeps the uncompacted run for the trajectory record, so
    apply() must hand back a new list."""
    g = SlidingGate(slot_chars=2000, stm_slots=2, turn_len=8)
    msgs = _msgs(12)
    before = len(msgs)
    g.apply(msgs, 0)
    assert len(msgs) == before, "apply() mutated the caller's list"


def test_window_oscillates_about_the_split():
    """The sine window is the point: live depth must go ABOVE and BELOW the
    0.5 mean, and must be smooth (no jump).

    This replaces an earlier monotonicity test. The gate used to be a ratchet
    that walked DOWN one slot per turn and never returned; that has been
    replaced by a standing oscillation, so monotonicity is deliberately gone.
    """
    g = SlidingGate(slot_chars=2000, stm_slots=4, turn_len=8, period=16)
    fr = [g.live_fraction(t) for t in range(16)]

    assert abs(sum(fr) / len(fr) - 0.5) < 0.02, f"mean not at the split: {sum(fr)/len(fr)}"
    assert min(fr) < 0.25, f"trough not reached: {min(fr)}"
    assert max(fr) > 0.75, f"crest not reached: {max(fr)}"
    # half the cycle sits BELOW the split -- the negative band
    below = sum(1 for v in fr if v < 0.5)
    assert below >= 6, f"too few turns below the split: {below}/16"
    # smooth: no turn-to-turn jump comparable to a sawtooth's 0 -> 1 step
    steps = [abs(fr[i + 1] - fr[i]) for i in range(len(fr) - 1)]
    assert max(steps) < 0.35, f"discontinuity in the window: {max(steps)}"


def test_live_depth_tracks_the_sine_not_the_floor():
    """REGRESSION. The floor and the sine originally scaled together (floor
    derived FROM the sine's keep), so the sine stopped controlling anything and
    window spread grew with the floor: 3090 chars at floor=4, 21132 at
    floor=16.

    The relationship is: live = max(floor, sine_target). So the floor must
    bind AT THE TROUGH and must NOT bind at the crest -- otherwise the sine is
    being overridden everywhere and the two are competing rather than composed.
    """
    g = SlidingGate(slot_chars=2000, stm_slots=4, turn_len=8,
                    period=16, min_live_msgs=6)
    asked = [g.live_messages(t, 20) for t in range(16)]

    floor = 6
    trough = min(range(16), key=lambda t: g.live_fraction(t))
    crest = max(range(16), key=lambda t: g.live_fraction(t))

    # at the trough the sine asks for almost nothing, so the floor must win
    assert asked[trough] == floor, (
        f"floor did not bind at the trough: asked {asked[trough]} != {floor}"
        f" (turn {trough}, live_frac {g.live_fraction(trough):.3f})")
    # at the crest the sine asks for much more, so the floor must NOT win
    assert asked[crest] > floor, (
        f"floor overrode the sine at the crest: asked {asked[crest]} <= {floor}")
    # and across the period the request must vary (i.e. the sine is in play)
    assert max(asked) - min(asked) >= 6, (
        f"live depth barely varies, the sine is not driving it: {asked}")


def test_apply_uses_the_sine_not_just_the_floor():
    """The unit test above pins live_messages(); this pins apply().

    A variant that made the floor the ONLY bound -- deleting the sine term from
    the max() inside apply() -- passed every other test here, because
    live_messages() itself was untouched. Only the WINDOW actually produced by
    apply() reveals it: without the sine the window sits at the floor forever
    instead of rising and falling.
    """
    g = SlidingGate(slot_chars=2000, stm_slots=4, turn_len=8,
                    period=16, min_live_msgs=6)
    msgs = _msgs(2)
    sizes = []
    for t in range(24):
        msgs = msgs + [
            {"role": "user", "content": f"turn {t}"},
            {"role": "assistant", "content": "z" * 3000},
        ]
        msgs = g.apply(msgs, t)
        sizes.append(sum(len(m.get("content") or "") for m in msgs))

    # the window must actually MOVE over the run, in step with the sine
    assert max(sizes) - min(sizes) > 3000, (
        f"window never oscillated, the sine is not reaching apply(): "
        f"spread {max(sizes) - min(sizes)}")
    # and it must still be bounded
    assert max(sizes) < 24 * 3000, f"window unbounded: {max(sizes)}"


def test_window_bounded_under_unbounded_growth():
    """REGRESSION. The window must stay bounded while the input grows without
    limit. Note it is NOT monotonic -- it oscillates -- so the property is a
    bound, not a direction."""
    g = SlidingGate(slot_chars=2000, stm_slots=4, turn_len=8, period=16)
    msgs = _msgs(4)
    sizes = []
    for t in range(48):
        msgs = msgs + [
            {"role": "user", "content": f"turn {t}"},
            {"role": "assistant", "content": "z" * 3000},
        ]
        msgs = g.apply(msgs, t)
        sizes.append(sum(len(m.get("content") or "") for m in msgs))
    unbounded = 48 * 3000 + 13000
    assert max(sizes) < unbounded / 2, (
        f"window not bounded: {max(sizes)} vs unbounded ~{unbounded}")
    assert len(g.retired) > 0, "gate never retired anything"


# ---- behaviour ------------------------------------------------------------

def test_state_channel_is_never_retired():
    """The harness state (H_t) is the continuity channel once history is
    elided. Compaction must not drop it."""
    secret = "obj_alpha: state=4821"
    g = SlidingGate(slot_chars=1500, stm_slots=2, turn_len=8)
    msgs = [{"role": "system", "content": "SYS"}]
    for t in range(12):
        msgs = msgs + [
            {"role": "user", "content": f"### Harness state (H_t)\n{secret}\nturn {t}"},
            {"role": "assistant", "content": "w" * 2500},
        ]
        msgs = g.apply(msgs, t)
        assert any(secret in (m.get("content") or "") for m in msgs), (
            f"H_t retired at turn {t}")


def test_retirement_is_the_xyz_output():
    """Retirement is the gate's XYZ output (t[13]-t[15]): the retired span must
    be written to ALL THREE axes, not just x.

    compactor.py wrote only "x" because it retired history as an
    undifferentiated blob. The XYZ lane's premise is that Belief / Progress /
    Experience are DIFFERENT coordinates; writing one axis collapses three
    coordinates into one.
    """
    calls = []

    class Buf:
        def store_axis(self, axis, text, persist=False):
            calls.append((axis, persist))
            return {"ok": True}

    g = SlidingGate(slot_chars=1500, stm_slots=2, turn_len=8, period=16)
    msgs = _msgs(10)
    for t in range(10):
        msgs = g.apply(msgs, t, buffer=Buf())

    assert calls, "nothing was offered to the buffer"
    axes = {a for a, _ in calls}
    assert axes == {"x", "y", "z"}, f"retirement did not reach all XYZ outputs: {axes}"
    assert all(p is False for _, p in calls), "retirement must not persist"
    assert g.stats()["axis_counts"]["z"] > 0, "z axis never written"
    # the audit trail records which axes each retirement touched
    assert all(r["axes"] == ["x", "y", "z"] for r in g.retired), (
        f"audit trail missing axes: {g.retired[:2]}")


def test_buffer_failure_is_fail_soft():
    """LTM is an optimisation, not a gate: a dead buffer must not stop
    compaction. (vectors.py is inert by default -- NEOH_LATENT=0 and its
    default base URL :8000 is down on this box.)"""

    class DeadBuf:
        def store_axis(self, *a, **k):
            raise RuntimeError("buffer down")

    g = SlidingGate(slot_chars=1500, stm_slots=2, turn_len=8, period=16)
    msgs = _msgs(10)
    for t in range(10):
        msgs = g.apply(msgs, t, buffer=DeadBuf())
    assert msgs, "dead buffer must not break compaction"
    assert g.stats()["n_retirements"] >= 0


def test_marker_says_what_was_retired():
    """law hand_cut_says: a cut item says by how much, and now which axes."""
    g = SlidingGate(slot_chars=1500, stm_slots=2, turn_len=8, period=16)
    msgs = _msgs(10)
    out = g.apply(msgs, 0)
    markers = [m for m in out if m["role"] == "user" and "retired" in m["content"]]
    if markers:
        assert "chars" in markers[0]["content"]
        assert "long-term" in markers[0]["content"]
        assert "live=" in markers[0]["content"], "marker must report the live fraction"


def test_sine_window_beats_threshold_sawtooth():
    """The property that matters against the OLD policy: bounded and smooth,
    where the threshold policy sawtooths across its trigger.

    Total chars carried is NOT the metric. A tight fit_chars lets the
    threshold policy win the total while still paying the cache cliff every
    cycle, so the assertion is on spread (smoothness) and on the bound.
    """

    def size(ms):
        return sum(len(m.get("content") or "") for m in ms)

    def threshold(msgs, fit_chars=24000, keep=6):
        out = list(msgs)
        while len(out) > 2 and size(out) > fit_chars:
            h = 0
            while h < len(out) and out[h]["role"] == "system":
                h += 1
            tail = max(h, len(out) - keep)
            if h >= tail:
                break
            out = out[:h] + [{"role": "user", "content": "[elided]"}] + out[tail:]
        return out

    g = SlidingGate(slot_chars=4000, stm_slots=3, turn_len=8, period=16)
    th = [{"role": "system", "content": "SYS" * 3000}]
    sl = [{"role": "system", "content": "SYS" * 3000}]
    th_sizes, sl_sizes = [], []
    for t in range(16):
        pair = [
            {"role": "user", "content": f"H_t turn {t}\n" + "q" * 200},
            {"role": "assistant", "content": "a" * 3000},
        ]
        th = threshold(th + pair)
        sl = g.apply(sl + pair, t)
        th_sizes.append(size(th))
        sl_sizes.append(size(sl))

    def spread(xs):
        return max(xs) - min(xs)

    assert spread(sl_sizes) <= spread(th_sizes), (
        f"sine {spread(sl_sizes)} not smoother than threshold {spread(th_sizes)}"
        f"\n  threshold: {th_sizes}\n  sine     : {sl_sizes}")
    # and the sine must stay bounded well below the uncompacted size
    assert max(sl_sizes) < 16 * 3000, f"sine window unbounded: {max(sl_sizes)}"
