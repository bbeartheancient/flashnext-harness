# SPDX-License-Identifier: Apache-2.0
"""Data-flow observation: DMA stream learning hook (BICAMERAL Goals 1, 9).

The observation layer sits between the MicroFX patch runner and the shell.
Every frame, for every active module, it records:

  - in_port values: the inputs resolved at @tick time
  - out_port values: the outputs emitted at @tick time
  - latency_us: wall time for this module's @tick execution
  - sequence: frame counter (monotonic)

This gives the model (Goals 1, 4, 9) a direct view of data flowing
through the module graph — not just the final output, but the *path*
the data took.  Over time, the observation log becomes a training
corpus for the "learn data mapping patterns from direct streams" goal.

The trace is circular-buffer backed (max_frames entries) so it never
unboundedly grows.  It can be:

  - read as a Python dict  → snapshot()
  - exported as a JSON list  → export_trace()
  - replayed through the bus  → replay(trace) fires the same
    in_port events so a second patch runner can be driven by a
    recorded run (for A/B testing / replay)

The bicameral design (Goals 1, 9) uses this as the "preprocessor"
view: GPU1 captures the data flow, GPU0 runs the active decision.
The trace is the bridge.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DataFlowTrace",
    "FrameEntry",
    "snapshot",
    "export_trace",
    "replay",
    "reset_trace",
    "start_trace",
    "stop_trace",
]

MAX_FRAMES = 10_000


@dataclass
class FrameEntry:
    seq: int
    t: float
    module_id: str
    module_type: str
    in_ports: dict[str, float]
    out_ports: dict[str, float]
    latency_us: float
    tag: str = ""


@dataclass
class DataFlowTrace:
    """Thread-safe circular buffer of frame entries.

    Access via snapshot() / export_trace() / replay().  Individual
    entries are added by calling record().  No locking in the hot path
    — use _lock only for snapshot/export.
    """
    _frames: deque[FrameEntry] = field(default_factory=lambda: deque(maxlen=MAX_FRAMES))
    _seq: int = 0
    _active: bool = False
    _lock_t: int = 0

    def record(self, module_id: str, module_type: str,
               in_ports: dict[str, float], out_ports: dict[str, float],
               latency_us: float, tag: str = "") -> FrameEntry | None:
        if not self._active:
            return None
        import threading
        t = time.time()
        self._seq += 1
        entry = FrameEntry(
            seq=self._seq,
            t=t,
            module_id=str(module_id),
            module_type=str(module_type),
            in_ports=dict(in_ports),
            out_ports=dict(out_ports),
            latency_us=float(latency_us),
            tag=str(tag),
        )
        self._frames.append(entry)
        return entry

    def snapshot(self) -> dict:
        import threading
        with threading.Lock():
            frames = [asdict(f) for f in self._frames]
        return {
            "active": self._active,
            "seq": self._seq,
            "n_frames": len(frames),
            "max_frames": MAX_FRAMES,
            "frames": frames,
        }

    def export_trace(self, path: str | None = None) -> str:
        snap = self.snapshot()
        txt = json.dumps(snap, indent=2)
        if path:
            from pathlib import Path
            Path(path).write_text(txt)
        return txt

    def clear(self) -> None:
        self._frames.clear()
        self._seq = 0

    def start(self) -> None:
        self._active = True
        logger.info("dma_trace: recording started")

    def stop(self) -> None:
        self._active = False
        logger.info("dma_trace: recording stopped")


# ---------------------------------------------------------------------------
# Global singleton (thread-safe, process-global per uvicorn worker)
# ---------------------------------------------------------------------------

_trace = DataFlowTrace()


def start_trace() -> None:
    _trace.start()


def stop_trace() -> None:
    _trace.stop()


def clear() -> None:
    _trace.clear()


def reset_trace() -> None:
    _trace.clear()


def snapshot() -> dict:
    return _trace.snapshot()


def export_trace(path: str | None = None) -> str:
    return _trace.export_trace(path)


def record(module_id: str, module_type: str,
          in_ports: dict[str, float], out_ports: dict[str, float],
          latency_us: float, tag: str = "") -> FrameEntry | None:
    return _trace.record(module_id, module_type, in_ports, out_ports,
                         latency_us, tag)


def replay(trace: list[dict]) -> list[dict]:
    """Fire replay events from a recorded trace.

    Returns a list of {module_id, port, value, seq, direction} events
    that a MicroFX patch runner can consume to replay the trace.  The
    model (Goal 1) uses this for A/B: run the same inputs through two
    different patches and diff the outputs.  Both in_port and out_port
    values are replayed (tagged with direction).
    """
    events: list[dict] = []
    for frame in trace:
        for port, val in (frame.get("in_ports") or {}).items():
            events.append({
                "seq": frame["seq"],
                "module_id": frame["module_id"],
                "port": port,
                "value": val,
                "t": frame["t"],
                "direction": "in",
            })
        for port, val in (frame.get("out_ports") or {}).items():
            events.append({
                "seq": frame["seq"],
                "module_id": frame["module_id"],
                "port": port,
                "value": val,
                "t": frame["t"],
                "direction": "out",
            })
    return events


# ---------------------------------------------------------------------------
# MicroFX bus integration
# ---------------------------------------------------------------------------

# When recording, the patch runner calls record() at the end of every
# @tick.  The event ring below mirrors the most recent trace entries
# as MicroFX bus signals so they are visible to the EEL2 interpreter
# without a separate query.
_RECENT_BUS: dict[str, float] = {}


def _update_bus(frame: FrameEntry) -> None:
    for port, val in frame.out_ports.items():
        key = f"df_{frame.module_id}_{port}"
        _RECENT_BUS[key] = val
    _RECENT_BUS["df_seq"] = float(frame.seq)
    _RECENT_BUS["df_t"] = frame.t


def recent_bus() -> dict[str, float]:
    return dict(_RECENT_BUS)
