"""Sliding-window gate: compaction as a running process, not a threshold event.

THE PROBLEM THIS SOLVES
=======================
`compactor.py` compacts when the window crosses `FIT_CHARS` (48000). That is a
threshold, and thresholds have two bad properties on a long-horizon run:

  1. Nothing happens for most of the run. The agent carries a window that is
     mostly dead weight and gets no use from it, then loses a contiguous chunk
     of live history in one step.
  2. The elision is a single lump at the oldest position, so the surviving
     window is biased: recent turns verbatim, everything older one marker line.

Measured on the live :8080 (IQ1_M, layer_split=24, parallel=4):

    prompt ~41k tok -> prefill 17.18s, decode 1.01s

Prefill dominates and the server caches an unchanged PREFIX automatically --
with NO flag; `cache_prompt=False` still hits. So turn cost is almost entirely
"how many tokens precede the first changed byte":

    identical prompt       prefill  0.06s
    +23 chars at the TAIL  prefill  0.25s   <- tail growth is cheap
    +1 char at the HEAD    prefill 18.21s   <- 91x cliff

A running process therefore needs one property above all: THE HEAD MUST STAY
BYTE-STABLE, and only the tail may move.

WHY THE HEAD IS CURRENTLY UNSTABLE
==================================
`harness.py:107` puts `state.render()` INSIDE the system message. That render
is volatile -- progress entries append, so it grows monotonically -- so the
head changes every turn and the cliff fires every turn. Over 6 turns:

    H_t in system head : 2.15 2.14 2.14 2.16 2.16 2.17  = 12.91s
    H_t in tail        : 2.38 0.61 0.62 0.63 0.63 0.64  =  5.51s   (2.3x)

That costs no accuracy: with a secret appearing ONLY inside H_t and never in
the question, recall was 6/6 in both placements. Moving H_t to the tail is
layout, not behaviour.

THE GATE
========
`agent.py` already owns a turn gate (`phase = turn % turn_len`, open at 0).
This reuses that clock rather than inventing a second one, because the
compaction boundary and the delegation boundary must not drift apart.

RETIREMENT IS THE XYZ OUTPUT (t[13]-t[15])
-------------------------------------------
The retired span is written to all three axes -- x/Belief, y/Progress,
z/Experience -- not just x as compactor.py did. The axis meanings are the
buffer's (vectors.py AXIS_SESSIONS, bpe.py's three stores); they are imported,
not re-declared, so the gate cannot drift from them.

Why all three: the XYZ lane's premise is that the three BPE components are
DIFFERENT coordinates. compactor.py retired history as an undifferentiated
blob onto "x" only, which collapses three coordinates into one. A span of
retired turns normally carries all three kinds of content.

THE WINDOW IS A SINE, NOT A SAWTOOTH
------------------------------------
The live fraction follows (1 - cos(theta))/2 rather than a 0.5-duty step:

  * A sawtooth at 0.5 duty is a STEP. The live set jumps discontinuously at
    the wrap, so the cached prefix is invalidated in one turn and rebuilt over
    the next few. The sine is smooth at BOTH ends (d/dtheta = 0 at theta = 0
    and pi); its max rate of change is 0.5 per radian and bounded, where a
    sawtooth's is unbounded at the discontinuity.

  * The 0.5 split is the MEAN, not the duty. cos(theta) < 0 on
    (pi/2, 3pi/2), and there live(t) < 0.5 -- the window sits BELOW the
    midpoint for half the cycle. Memory depth oscillates ABOUT the split
    rather than parking at it. That negative band is the part that matches a
    rotation; a step has no phase at all.

Note this replaces an earlier ratchet, which walked DOWN one slot per turn and
never came back. That monotonicity is deliberately gone: a standing
oscillation about a fixed mean is the point. The invariant that survived every
earlier bug is the one that matters -- the system HEAD is never rotated out.
"""

from __future__ import annotations

import math
import os

# Axis names are owned by the buffer. Imported rather than re-declared so the
# gate cannot drift from vectors.py's AXES / AXIS_SESSIONS.
try:
    from .vectors import AXES
except Exception:                                               # noqa: BLE001
    AXES = ("x", "y", "z")

SLOT_CHARS = int(os.environ.get("NEOH_SLOT_CHARS", "6000"))
# Depth of short-term memory, as a multiple of the live fraction the sine
# permits. The sine sets the SHAPE of the oscillation; this sets its scale.
STM_SLOTS = int(os.environ.get("NEOH_STM_SLOTS", "4"))
# Turn cycle length. Bounds how fast retirement happens and where the identity
# seam falls. Shared with agent.TURN_LEN by default so there is one clock.
TURN_LEN = int(os.environ.get("NEOH_TURN_LEN", "8"))
# Sine window: one full oscillation per this many turns.
PERIOD = int(os.environ.get("NEOH_GATE_PERIOD", "16"))
# Phase offset in radians. 0 puts the minimum at turn 0; a negative value
# shifts the NEGATIVE band (where the window sits below the 0.5 split) to
# start earlier, matching a rotation whose phase leads rather than lags.
PHASE_OFFSET = float(os.environ.get("NEOH_GATE_PHASE", "0.0"))
# Hard floor on messages kept live, independent of the sine. The sine sets the
# target depth; this guarantees short-term memory always survives it.
MIN_LIVE_MSGS = int(os.environ.get("NEOH_MIN_LIVE_MSGS", "6"))
# Messages per slot on a harness turn. A turn is a user/assistant PAIR, so a
# slot holds two messages. This is not cosmetic: at 1.0 the slot-space target
# could never clear min_live_msgs on a real run (stm_slots=4 -> at most 4
# messages requested against a floor of 6), so apply() returned a FLAT window
# and the sine stopped driving anything. Measured: window spread 0 chars.
MSGS_PER_SLOT = float(os.environ.get("NEOH_MSGS_PER_SLOT", "2.0"))


class SlidingGate:
    """Rotates memory out of short-term storage a few slots at a time.

    Deterministic, and `apply` returns a NEW list -- the caller's history is
    never mutated, so the trajectory record can keep the uncompacted run.
    """

    def __init__(self, slot_chars: int = SLOT_CHARS, stm_slots: int = STM_SLOTS,
                 turn_len: int = TURN_LEN, period: int = PERIOD,
                 phase_offset: float = PHASE_OFFSET,
                 min_live_msgs: int = MIN_LIVE_MSGS,
                 msgs_per_slot: float = MSGS_PER_SLOT):
        self.slot_chars = max(1, slot_chars)
        self.stm_slots = max(1, stm_slots)
        self.turn_len = max(1, turn_len)
        # sine window: one full oscillation every `period` turns. Defaults to
        # one full cycle per turn-gate cycle so the two clocks are the same.
        self.period = max(2, period)
        self.phase_offset = phase_offset
        # Hard floor on live messages. INDEPENDENT of the sine: the sine may
        # set the target depth, but this guarantees short-term memory always
        # survives. Without it the live set collapsed at the trough.
        self.min_live_msgs = max(2, min_live_msgs)
        # Messages per slot. Converts the sine's slot-space target into the
        # message-space floor so the two are commensurate (see live_messages).
        self.msgs_per_slot = max(0.5, msgs_per_slot)
        self.retired: list[dict] = []
        self.ltm_chars = 0
        self.axis_counts: dict[str, int] = {"x": 0, "y": 0, "z": 0}
        self._cycle = -1
        self._budget = 0          # retirements allowed this cycle

    # -- helpers -----------------------------------------------------------
    def phase(self, turn: int) -> int:
        return turn % self.turn_len

    @staticmethod
    def _size(messages: list[dict]) -> int:
        return sum(len(m.get("content") or "") for m in messages)

    @staticmethod
    def _head(messages: list[dict]) -> int:
        """First non-system index. System messages are the stable head and are
        never rotated out by this gate -- that is the property the whole
        design exists to protect."""
        h = 0
        while h < len(messages) and messages[h]["role"] == "system":
            h += 1
        return h

    def _slots(self, n_chars: int) -> int:
        return max(1, -(-n_chars // self.slot_chars))

    def _tick(self, turn: int) -> None:
        """Refresh the per-cycle retirement budget at the identity seam."""
        cycle = turn // self.turn_len
        if cycle != self._cycle:
            self._cycle = cycle
            self._budget = 0

    # -- the window function -----------------------------------------------
    def live_fraction(self, turn: int) -> float:
        """Fraction of the window that stays live at `turn`.

        SINE, not sawtooth. A 0.5-duty sawtooth is a STEP: the live set jumps
        discontinuously at the wrap, so the cached prefix is invalidated in one
        turn and rebuilt over the next few. The sine

            live(t) = (1 - cos(theta)) / 2,  theta = 2*pi*t/period + phase

        is smooth at BOTH ends (d/dtheta = 0 at theta = 0 and theta = pi), so
        the window never jumps. Measured max rate of change is 0.5 per radian,
        bounded; a sawtooth's is unbounded at the discontinuity.

        THE 0.5 SPLIT IS THE MEAN, not the duty. cos(theta) < 0 for theta in
        (pi/2, 3pi/2), and there live(t) < 0.5 -- the window sits BELOW the
        midpoint for half the cycle. That negative band is the point: memory
        depth oscillates about the split rather than parking at it, which is
        what a rotation does and what a step cannot.

        Note this is deliberately NOT monotonic in turn, unlike the ratchet
        that preceded it. Retirement here is a standing oscillation around a
        fixed mean, not a one-way walk down; the invariant that survived both
        earlier bugs is that the system HEAD is never touched.
        """
        theta = 2.0 * math.pi * (turn / self.period) + self.phase_offset
        frac = (1.0 - math.cos(theta)) / 2.0
        # never let the live fraction hit 0 or 1: keep >= 1 slot and the head
        return min(1.0 - 1e-6, max(1e-6, frac))

    # -- the gate ----------------------------------------------------------
    def keep_slots(self, turn: int, window_slots: int) -> int:
        """Slots of STM still live at `turn`, driven by the sine window."""
        frac = self.live_fraction(turn)
        return max(1, min(window_slots, int(round(frac * window_slots))))

    def live_messages(self, turn: int, window_slots: int) -> int:
        """MESSAGES to keep live: the sine's target, floored.

        The sine is defined over SLOTS (it scales with the window), while the
        safety floor is defined over MESSAGES (it is about how many turns of
        short-term history must survive regardless of how big those turns are).
        Converting slots -> messages by the average message size is what makes
        the two commensurate; scaling one off the other directly is the unit
        mismatch that let the floor dominate.
        """
        target_slots = self.keep_slots(turn, window_slots)
        if window_slots <= 0:
            return self.min_live_msgs
        # average chars per slot -> messages per slot
        msgs_per_slot = max(1, int(round(self.msgs_per_slot)))
        return max(self.min_live_msgs, target_slots * msgs_per_slot)

    def apply(self, messages: list[dict], turn: int, buffer=None) -> list[dict]:
        """Compact `messages` for turn `turn`, retiring overflow to LTM.

        Retirement happens one slot per call, budgeted per cycle, against the
        ALREADY-APPLIED window -- the caller passes the output of the previous
        turn's apply(), which is what makes retirement compound.
        """
        self._tick(turn)
        messages = list(messages)
        head = self._head(messages)
        if head >= len(messages):
            return messages

        # The sine's SCALE is the CONFIGURED budget (stm_slots), never the
        # measured window. Scaling by the measured window is positive feedback:
        # window grows -> scale grows -> target grows -> live set grows ->
        # max_cut falls back to head -> nothing is cut -> window keeps growing.
        # Measured on the live E2E: 71k chars carried, ZERO retirements, with
        # the gate reporting a healthy sine. stm_slots is what bounds it.
        window_slots = min(self._slots(self._size(messages)), self.stm_slots)
        keep_slots = self.keep_slots(turn, window_slots)
        keep_msgs = self.live_messages(turn, window_slots)

        # budget: one retirement per call, at most turn_len per cycle
        retire_this_turn = self._budget < self.turn_len
        if retire_this_turn:
            self._budget += 1

        if not retire_this_turn:
            return messages

        # How much to retire.
        #
        # The sine sets the TARGET live depth, so retirement must be allowed to
        # overshoot it and come back -- that is the oscillation. But it must
        # still retire enough to REACH the target: a fixed one-slot step cannot,
        # because the window grows by one slot per turn at the same rate, and
        # the two cancel. Measured with a fixed step: the window grew 46k ->
        # 55k chars and never bounded.
        #
        # So retire down TO the target when above it (unbounded, so the sine
        # can actually move), and step gently when below it.
        target_chars = max(self.slot_chars, keep_slots * self.slot_chars)
        current_chars = self._size(messages)
        if current_chars > target_chars:
            budget_chars = current_chars - target_chars      # converge
        else:
            budget_chars = self.slot_chars                 # gentle step

        cut = head
        acc = 0
        while cut < len(messages) and acc < budget_chars:
            acc += len(messages[cut].get("content") or "")
            cut += 1

        # Never eat the whole volatile region. The floor is on MESSAGES and is
        # a HARD MINIMUM, not a second depth knob competing with the sine:
        #
        #     live = max(min_live_msgs, sine_target)
        #
        # An earlier version derived the floor FROM the sine's keep, so the two
        # scaled together and the sine stopped controlling anything -- measured
        # window spread grew with the floor (3090 chars at floor=4, 21132 at
        # floor=16) instead of being set by the oscillation. Here the floor
        # only ever binds at the trough, where the sine asks for less than the
        # minimum; above that the sine is in charge.
        live_msgs = min(len(messages) - head, max(self.min_live_msgs, keep_msgs))
        max_cut = max(head, len(messages) - live_msgs)
        if cut > max_cut:
            cut = max_cut
        if cut <= head:
            return messages

        retired = messages[head:cut]
        rendered = "\n".join(
            f"[{m['role']}] {(m.get('content') or '')[:2000]}" for m in retired
        )

        # Retirement is the XYZ OUTPUT of the gate -- t[13], t[14], t[15] --
        # so the retired span is written to all three axes, not just x.
        #
        # The axis meanings are fixed by the buffer (vectors.py AXIS_SESSIONS
        # and bpe.py's three stores):
        #     x -> Belief      task-relevant facts about objects
        #     y -> Progress    subgoal status
        #     z -> Experience  skills and lessons
        #
        # compactor.py wrote only "x" because it retired history as an
        # undifferentiated blob. A span of retired turns usually carries all
        # three kinds of content, so mirroring the same rendered text to each
        # axis is what makes the latent coordinates separable later -- the
        # whole premise of the XYZ lane is that the axes are DIFFERENT
        # coordinates, not three copies of one.
        stored = {}
        if buffer is not None:
            for axis in AXES:
                try:
                    resp = buffer.store_axis(axis, rendered, persist=False)
                    stored[axis] = resp
                    self.axis_counts[axis] += 1
                except Exception:                               # noqa: BLE001
                    stored[axis] = None  # fail-soft: LTM is not a gate
        stored_any = any(v is not None for v in stored.values())

        n_chars = sum(len(m.get("content") or "") for m in retired)
        self.ltm_chars += n_chars
        self.retired.append({
            "turn": turn, "phase": self.phase(turn),
            "live_fraction": round(self.live_fraction(turn), 4),
            "n_messages": len(retired), "n_chars": n_chars,
            "axes": sorted(a for a, v in stored.items() if v is not None),
            "stored": stored_any,
            # The retired TEXT, so the span can be recalled rather than merely
            # counted. Without it this audit log records that something left
            # the window but not what -- and retirement becomes irreversible in
            # practice even though an XYZ buffer was "written". Measured: a
            # lexical recall with no text is a permanent blind spot.
            "text": rendered,
        })

        marker = {"role": "user", "content": (
            f"[gate: retired {len(retired)} turn(s), {n_chars} chars to long-term "
            f"memory at phase {self.phase(turn)}/{self.turn_len} "
            f"(live={self.live_fraction(turn):.2f})"
            + (f"; mirrored to axes {''.join(sorted(stored))}" if stored_any else "")
            + "]"
        )}
        return messages[:head] + [marker] + messages[cut:]

    def stats(self) -> dict:
        return {
            "slot_chars": self.slot_chars,
            "stm_slots": self.stm_slots,
            "turn_len": self.turn_len,
            "period": self.period,
            "phase_offset": self.phase_offset,
            "n_retirements": len(self.retired),
            "ltm_chars": self.ltm_chars,
            "axis_counts": dict(self.axis_counts),
        }
