"""Pipecat voice-leg probe.

Harness composition is pipecat (realtime front) + Pydantic AI + duty LLM.
The voice pipeline is last of Track 2 — this module reports that honestly
and lists the functions the slash menu / duty agent can already name.
"""

from __future__ import annotations

import importlib.util
import os
import socket


def _port_up(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(0.35)
            return s.connect_ex((host, int(port))) == 0
    except OSError:
        return False


def pipecat_status() -> dict:
    """Voice-leg status. Installed/running flags; no fake pipeline."""
    # A bare ~/pipecat checkout resolves as a NAMESPACE package
    # (loader=None) once any sys.path entry exposes $HOME — that
    # is NOT an install. Only a real module loader counts.
    spec = importlib.util.find_spec("pipecat")
    installed = spec is not None and getattr(spec, "loader", None) is not None
    version = None
    if installed:
        try:
            import pipecat  # type: ignore

            version = getattr(pipecat, "__version__", None)
        except Exception:
            version = None
    port = int(os.environ.get("FABRIC_PIPECAT_PORT", "8100"))
    running = _port_up(port)
    return {
        "lane": "pipecat",
        "stage": "wired-status" if installed or running else "concept",
        "installed": installed,
        "version": version,
        "running": running,
        "port": port,
        "functions": [
            {"id": "wake", "wired": False, "note": "edge OpenWakeWord"},
            {"id": "barge_in", "wired": False, "note": "VAD stops Piper"},
            {"id": "transcribe", "wired": False, "note": "Whisper utterance"},
            {"id": "speak", "wired": False, "note": "Piper TTS"},
        ],
        "next": "Track 2 voice leg after HA reconnect; do not run Whisper 24/7 on a B70",
        "duty": "text/chat is the daily driver until the WS leg lands",
    }
