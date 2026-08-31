# SPDX-License-Identifier: Apache-2.0
"""OpenCode → vLLM proxy shim with n-gram live-buffer feed.

OpenCode's duty provider points its baseURL at ``/opencode/v1`` here
instead of vLLM directly.  Every chat-completions request is forwarded
verbatim (streaming included); around the forward the shim:

  1. ACTIVATES the "opencode" session buffer before the prefill, so the
     request sees the rolling edge table (the KV-window extension).
  2. After the reply completes, feeds the turn's tail (last user
     message only, FABRIC_NGRAM_LIVE_CHARS-bounded — the assistant
     reply is excluded to avoid a self-referential decode echo) into
     that buffer via latent.live_note — fire-and-forget, never blocks
     the response.

The capture span is only the tail (a few hundred tokens), NOT the full
re-sent context — feeding the whole window per request would cost a
full extra prefill per turn.  Older turns are already in the buffer
from their own tails; this is the incremental rolling-feed pattern.

Compaction transfer: OpenCode compacts CLIENT-side — when its token
counter trips, it sends a summarize request (system prompt "You are a
context summarization agent …") whose messages carry the FULL
pre-compaction conversation, then replaces its history with the
summary.  The shim detects that request and, BEFORE forwarding it,
synchronously transfers the rendered full conversation into the
opencode session buffer (live lane, merge).  The engine applies the
refresh before the next forward, so the post-compaction request sees
the elided history as a layer-63 edge delta — the same contract as the
duty agent's FABRIC_NGRAM_COMPACT lane.  ``FABRIC_OPENCODE_COMPACT=0``
disables; budget = FABRIC_OPENCODE_COMPACT_CHARS (default 48000).

Embed leg (FABRIC_OPENCODE_EMBED=1): the same compaction request ALSO
compacts the rendered conversation into a dct_topk prompt_embeds block
(latent.embed_transfer → latent.EMBED_BLOCKS, newest-wins on disk).
Subsequent non-compaction requests get the active block spliced in as
a prompt_embeds user message before the last user message
(``_splice_embed``) — the elided history rides as a soft prompt, not
just an edge delta.  When the embed budget
(FABRIC_OPENCODE_EMBED_CHARS) matches the compact budget the hidden
capture is shared between both legs; rate/gate/rows/TTL are
FABRIC_OPENCODE_EMBED_{RATE,GATE,MAX_ROWS,TTL_H}.

Session identity: OpenCode sends no session header, so all traffic
shares the ``opencode`` session (single-operator duty; env
FABRIC_OPENCODE_SESSION overrides).
"""
from __future__ import annotations

import json
import threading
import urllib.request

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import bpe_bridge, latent

router = APIRouter()

_TIMEOUT_S = 900.0


def _session() -> str:
    import os
    return os.environ.get("FABRIC_OPENCODE_SESSION", "opencode")


def _upstream(path: str) -> str:
    return latent._llm_url() + path


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) for p in c
                         if isinstance(p, dict))
    return ""


def _last_user_text(body: dict) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") != "user":
            continue
        t = _msg_text(m)
        if t:
            return t
    return ""


_COMPACT_SYSTEM = "context summarization agent"
_COMPACT_USER = "Summarize what was done in this conversation"


def _is_compaction(body: dict) -> bool:
    """OpenCode's summarize request: dedicated summarizer system prompt,
    or the summarize instruction as the last user message."""
    msgs = body.get("messages") or []
    for m in msgs:
        if m.get("role") == "system" and _COMPACT_SYSTEM in _msg_text(m):
            return True
    last_user = _last_user_text(body)
    return _COMPACT_USER in last_user


def _render_conversation(body: dict, budget: int) -> str:
    """Flatten all messages (the full pre-compaction history) into a
    role-labelled plain-text span for hidden capture."""
    parts: list[str] = []
    for m in body.get("messages") or []:
        role = str(m.get("role") or "?")
        text = _msg_text(m).strip()
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            text += (f"\n[tool_call {fn.get('name', '?')}] "
                     f"{str(fn.get('arguments', ''))[:2000]}")
        if text.strip():
            parts.append(f"{role.upper()}: {text.strip()}")
    return "\n\n".join(parts)[:budget]


def _splice_embed(body: dict, b64: str) -> dict:
    """Return a copy of ``body`` with a prompt_embeds user message
    inserted before the last user message (the elided history as a
    soft prompt).  Returns ``body`` unchanged when there is no user
    message.  Never mutates ``body``."""
    msgs = body.get("messages") or []
    idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            idx = i
            break
    if idx < 0:
        return body
    out = dict(body)
    out["messages"] = list(msgs)
    out["messages"].insert(idx, {
        "role": "user",
        "content": [{"type": "prompt_embeds", "data": b64}],
    })
    return out


def _compaction_transfer(body: dict) -> None:
    """Synchronous full-history transfer on a detected compaction
    request.  Blocking on purpose: the refresh (and embed block) must
    be published before OpenCode's post-compaction prefill arrives.
    Two legs: the n-gram live-lane merge (FABRIC_OPENCODE_COMPACT) and
    the prompt_embeds block (FABRIC_OPENCODE_EMBED); when the budgets
    match they share one hidden capture."""
    import os
    ngram_on = (latent._ngram_live_enabled()
                and os.environ.get("FABRIC_OPENCODE_COMPACT", "1") == "1")
    embed_on = latent._embed_enabled()
    if not (ngram_on or embed_on):
        return
    budget = int(os.environ.get("FABRIC_OPENCODE_COMPACT_CHARS", "48000"))
    span = _render_conversation(body, budget)
    ebudget = latent._embed_budget() if embed_on else 0
    espan = (span if ebudget == budget
             else _render_conversation(body, ebudget)) if embed_on else ""
    if not span.strip() and not espan.strip():
        return
    arr = None
    if span.strip() and espan == span:
        # Budgets match: one capture serves both legs.
        arr, _why = latent._capture_hidden_states(span, max_chars=budget)
    if ngram_on and span.strip():
        try:
            if arr is not None:
                latent.live_note_arr(_session(), span, arr,
                                     max_chars=budget)
            else:
                latent.live_note(_session(), span, max_chars=budget)
        except Exception:  # noqa: BLE001
            pass
    if embed_on:
        try:
            if arr is not None:
                latent.embed_transfer_capture(_session(), arr)
            else:
                latent.embed_transfer(_session(), espan)
        except Exception:  # noqa: BLE001
            pass


def _feed(user_text: str, reply: str) -> None:
    """Fire-and-forget tail feed into the opencode session buffer.

    User text only: storing the assistant's own reply created a
    self-referential echo — live edge rows are captured from the exact
    tokens the model then re-decodes, so the layer-63 delta pulled
    generation back onto its own recent output (intra-sentence
    stutter, "the the the"), amplified by frequency pinning.
    """
    if not latent._ngram_live_enabled() or not user_text.strip():
        return

    def _work() -> None:
        try:
            latent.live_note(_session(), f"User: {user_text}")
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_work, daemon=True).start()


def _activate() -> None:
    if latent._ngram_live_enabled():
        try:
            latent._PLE_BUFFER.activate(_session())
        except Exception:  # noqa: BLE001
            pass


@router.get("/opencode/v1/models")
def oc_models():
    with urllib.request.urlopen(_upstream("/models"),
                                timeout=30) as r:
        return JSONResponse(json.loads(r.read().decode()))


@router.post("/opencode/v1/chat/completions")
async def oc_chat(req: Request):
    import copy
    body = await req.json()
    orig_body = copy.deepcopy(body)
    compaction = _is_compaction(body)
    bpe_on = bpe_bridge.enabled()
    if compaction and bpe_on:
        bpe_bridge.mark_compaction(_session())
    if not compaction and latent._embed_enabled():
        blk = latent.EMBED_BLOCKS.active(_session())
        if blk is not None:
            body = _splice_embed(body, blk["b64"])
    if not compaction and bpe_on:
        body = bpe_bridge.inject(body, _session())
    raw = json.dumps(body).encode()
    user_text = _last_user_text(orig_body)
    if compaction:
        _compaction_transfer(orig_body)
    _activate()
    upstream_req = urllib.request.Request(
        _upstream("/chat/completions"), data=raw,
        headers={"Content-Type": "application/json",
                 "User-Agent": "woodfire-fabric-ocproxy"})

    if not body.get("stream"):
        with urllib.request.urlopen(upstream_req,
                                    timeout=_TIMEOUT_S) as r:
            payload = json.loads(r.read().decode())
        reply = ""
        try:
            reply = str(payload["choices"][0]["message"].get("content")
                        or "")
        except Exception:  # noqa: BLE001
            pass
        if not compaction:
            _feed(user_text, reply)
            if bpe_on:
                try:
                    bpe_bridge.apply_reply(reply, _session())
                except Exception:  # noqa: BLE001
                    pass
        return JSONResponse(payload)

    def relay():
        parts: list[str] = []
        with urllib.request.urlopen(upstream_req,
                                    timeout=_TIMEOUT_S) as r:
            for chunk in r:
                yield chunk
                # accumulate assistant deltas for the tail feed
                for line in chunk.decode("utf-8", "ignore").splitlines():
                    if not line.startswith("data:") or "[DONE]" in line:
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                        delta = (ev.get("choices") or [{}])[0].get(
                            "delta") or {}
                        if delta.get("content"):
                            parts.append(str(delta["content"]))
                    except Exception:  # noqa: BLE001
                        pass
        if not compaction:
            reply_text = "".join(parts)
            _feed(user_text, reply_text)
            if bpe_on:
                try:
                    bpe_bridge.apply_reply(reply_text, _session())
                except Exception:  # noqa: BLE001
                    pass

    return StreamingResponse(
        relay(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"})
