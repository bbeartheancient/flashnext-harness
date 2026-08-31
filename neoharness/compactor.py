"""Context compaction for long-horizon runs.

When the message window crosses NEOH_FIT_CHARS, the oldest non-system turns
beyond NEOH_KEEP_TURNS are elided — but before they're dropped, the elided
span is rendered and offered to the XYZ latent lane
(``XYZBuffer.store_axis("x", ...)``, persist=False): the M1
FABRIC_NGRAM_COMPACT contract — elided history rides the n-gram edge buffer
instead of a token summary. With the latent lane off, compaction degrades to
a plain one-line elision note.

Never elides: system messages, and the last KEEP_TURNS messages.
"""

from __future__ import annotations

import os

FIT_CHARS = int(os.environ.get("NEOH_FIT_CHARS", "48000"))
KEEP_TURNS = int(os.environ.get("NEOH_KEEP_TURNS", "6"))


class ContextCompactor:
    def __init__(self, buffer=None, fit_chars: int = FIT_CHARS,
                 keep_turns: int = KEEP_TURNS):
        self.buffer = buffer  # XYZBuffer or None
        self.fit_chars = fit_chars
        self.keep_turns = keep_turns
        self.transfers: list[dict] = []  # audit log of latent transfers

    @staticmethod
    def _size(messages: list[dict]) -> int:
        return sum(len(m.get("content") or "") for m in messages)

    def fit(self, messages: list[dict]) -> list[dict]:
        """Return messages shrunk under the budget (in-place safe copy)."""
        messages = list(messages)
        while len(messages) > 2 and self._size(messages) > self.fit_chars:
            before = self._size(messages)
            # First elidable index: after system messages, leaving keep_turns
            # messages at the tail.
            head = 0
            while head < len(messages) and messages[head]["role"] == "system":
                head += 1
            tail_start = max(head, len(messages) - self.keep_turns)
            if head >= tail_start:
                break  # nothing elidable without touching the kept tail
            elided = messages[head:tail_start]
            note = self._elide(elided)
            messages = messages[:head] + [note] + messages[tail_start:]
            if self._size(messages) >= before:
                break  # elision is not shrinking the window; stop
        return messages

    def _elide(self, elided: list[dict]) -> dict:
        rendered = "\n".join(
            f"[{m['role']}] {(m.get('content') or '')[:2000]}" for m in elided
        )
        n_chars = sum(len(m.get("content") or "") for m in elided)
        transfer = None
        if self.buffer is not None:
            transfer = self.buffer.store_axis("x", rendered, persist=False)
        self.transfers.append({
            "n_messages": len(elided), "n_chars": n_chars,
            "stored": transfer is not None,
        })
        marker = (
            f"[harness: elided {len(elided)} earlier turns ({n_chars} chars)"
            + ("; transferred to n-gram buffer" if transfer else "")
            + "]"
        )
        return {"role": "user", "content": marker}
