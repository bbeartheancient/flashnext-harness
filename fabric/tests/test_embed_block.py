# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the opencode shim's embed leg: EmbedBlockStore
(newest-wins rolling store with TTL) and _splice_embed (prompt_embeds
insertion), plus the embed_transfer ladder/gate (GPU lane monkeypatched
— no worker, no engine)."""

import time

import numpy as np
import pytest

from fabric import latent
from fabric.opencode_proxy import _splice_embed


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_EMBED_BLOCK_DIR", str(tmp_path))
    return latent.EmbedBlockStore(tmp_path)


def test_save_active_roundtrip(store):
    rec = store.save("s1", "QUJD", 12, {"row_cos": 0.95})
    assert rec["rows"] == 12
    blk = store.active("s1")
    assert blk is not None
    assert blk["b64"] == "QUJD"
    assert blk["rows"] == 12
    assert blk["row_cos"] == 0.95
    assert store.active("other") is None


def test_newest_wins(store):
    store.save("s1", "QUJD", 12)
    store.save("s1", "REVG", 34)
    blk = store.active("s1")
    assert blk["b64"] == "REVG"
    assert blk["rows"] == 34
    assert len(list(store._dir.glob("s1.json"))) == 1


def test_ttl_expiry_reaps(store):
    store.save("s1", "QUJD", 12)
    p = store._path("s1")
    import json as _json
    rec = _json.loads(p.read_text())
    rec["ts"] = time.time() - 2 * 3600  # 2 h old
    p.write_text(_json.dumps(rec))
    assert store.active("s1", ttl_h=1.0) is None
    assert not p.exists()              # reaped on read
    # ttl 0 disables expiry
    store.save("s2", "QUJD", 1)
    rec = _json.loads(store._path("s2").read_text())
    rec["ts"] = time.time() - 100 * 3600
    store._path("s2").write_text(_json.dumps(rec))
    assert store.active("s2", ttl_h=0) is not None


def test_clear(store):
    store.save("s1", "QUJD", 12)
    assert store.clear("s1") is True
    assert store.active("s1") is None
    assert store.clear("s1") is False  # already gone


_BODY = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "last"},
    ],
}


def test_splice_embed_inserts_before_last_user():
    out = _splice_embed(_BODY, "QkX=")
    msgs = out["messages"]
    assert len(msgs) == len(_BODY["messages"]) + 1
    ins = msgs[3]
    assert ins["role"] == "user"
    assert ins["content"] == [{"type": "prompt_embeds", "data": "QkX="}]
    assert msgs[4]["content"] == "last"
    # original untouched (no mutation)
    assert len(_BODY["messages"]) == 4
    assert out is not _BODY
    assert out["messages"] is not _BODY["messages"]


def test_splice_embed_no_user_is_noop():
    body = {"messages": [{"role": "system", "content": "sys"}]}
    assert _splice_embed(body, "QkX=") is body
    assert _splice_embed({}, "QkX=") == {}


def test_splice_embed_only_copies_top_level():
    out = _splice_embed(_BODY, "QkX=")
    assert out["model"] == "m"
    # nested message dicts are shared (shallow), only the list is new
    assert out["messages"][0] is _BODY["messages"][0]


class _FakeGpu:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def compact_b64(self, b64, shape, rate):
        self.calls.append((b64, tuple(shape), rate))
        return self.payload


def test_embed_transfer_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("FABRIC_OPENCODE_EMBED", "1")
    monkeypatch.setenv("FABRIC_EMBED_BLOCK_DIR", str(tmp_path))
    monkeypatch.setattr(latent, "EMBED_BLOCKS", latent.EmbedBlockStore(tmp_path))
    fake = _FakeGpu({"b64": "QkxP", "rows": 8, "effective_rate": 0.5,
                     "row_cos": 0.9})
    monkeypatch.setattr(latent, "_GPU", fake)
    arr = np.zeros((4, 5120), dtype="<f2")
    rec = latent.embed_transfer_capture("opencode", arr)
    assert rec["rows"] == 8
    assert fake.calls and fake.calls[0][1] == (4, 5120)
    blk = latent.EMBED_BLOCKS.active("opencode")
    assert blk["b64"] == "QkxP"


def test_embed_transfer_disabled(monkeypatch):
    monkeypatch.delenv("FABRIC_OPENCODE_EMBED", raising=False)
    assert latent.embed_transfer("s", "some span") is None


def test_embed_transfer_ladder_and_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("FABRIC_OPENCODE_EMBED", "1")
    monkeypatch.setenv("FABRIC_OPENCODE_EMBED_GATE", "0.8")
    monkeypatch.setenv("FABRIC_OPENCODE_EMBED_MAX_ROWS", "16")
    monkeypatch.setattr(latent, "EMBED_BLOCKS", latent.EmbedBlockStore(tmp_path))
    arr = np.zeros((4, 8), dtype="<f2")

    # lane down
    monkeypatch.setattr(latent, "_GPU", _FakeGpu(None))
    assert "error" in latent.embed_transfer_capture("s", arr)

    # over max rows
    monkeypatch.setattr(latent, "_GPU",
                        _FakeGpu({"b64": "x", "rows": 99, "row_cos": 1.0}))
    assert "bad rows" in latent.embed_transfer_capture("s", arr)["error"]

    # below the row_cos gate
    monkeypatch.setattr(latent, "_GPU",
                        _FakeGpu({"b64": "x", "rows": 8, "row_cos": 0.5}))
    assert "below gate" in latent.embed_transfer_capture("s", arr)["error"]

    # passes gate at exactly the threshold
    monkeypatch.setattr(latent, "_GPU",
                        _FakeGpu({"b64": "x", "rows": 8, "row_cos": 0.8}))
    rec = latent.embed_transfer_capture("s", arr)
    assert rec["rows"] == 8
