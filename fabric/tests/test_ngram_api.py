# SPDX-License-Identifier: Apache-2.0
"""Tests for the n-gram PLE HTTP bridge (fabric/ngram_api.py + routes)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("memvid_sdk", reason="memvid-sdk not installed")

_BIC = str(Path(__file__).resolve().parents[2] / "python" / "afi" / "bicameral")
if _BIC not in sys.path:
    sys.path.insert(0, _BIC)

import writable_ngram as wn  # noqa: E402
from fabric import ngram_api  # noqa: E402


@pytest.fixture(autouse=True)
def shard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NGRAM_SHARD_DIR", str(tmp_path / "shards"))
    yield
    wn.close_all()


def _vec(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(160).astype(np.float16)


def test_store_then_lookup_roundtrip():
    v = _vec(0)
    r = ngram_api.store_rows(3, [{"token": 42, "vector": v.tolist()}])
    assert r == {"stored": 1, "layer": 3}
    ngram_api.flush()
    out = ngram_api.lookup(3, 42)
    assert out["found"] is True
    assert out["write_count"] == 1
    np.testing.assert_allclose(out["vector"], v.astype(np.float32),
                               atol=1e-3)


def test_lookup_miss():
    assert ngram_api.lookup(7, 999999) == {"found": False}


def test_lookup_batch_hits_only():
    ngram_api.store_rows(3, [{"token": 10, "vector": _vec(1).tolist()},
                             {"token": 11, "vector": _vec(2).tolist()}])
    ngram_api.flush()
    out = ngram_api.lookup_batch(3, [10, 11, 12])
    assert out["found"] == 2
    assert set(out["vectors"]) == {"10", "11"}


def test_store_rejects_bad_vector():
    r = ngram_api.store_rows(3, [{"token": 5, "vector": [1.0, 2.0]}])
    assert "error" in r
    r = ngram_api.store_rows(3, [{"token": 5}])
    assert "error" in r


def test_stats_reports_open_layers():
    ngram_api.lookup(3, 1)
    ngram_api.lookup(7, 1)
    st = ngram_api.stats()
    assert st["open_layers"] == [3, 7]


def test_routes_registered():
    from fastapi.testclient import TestClient
    from fabric.main import app
    client = TestClient(app)
    r = client.get("/api/ngram/lookup", params={"layer": 3, "token": 123})
    assert r.status_code == 200 and r.json() == {"found": False}
    r = client.post("/api/ngram/store",
                    json={"layer": 3, "token": 123,
                          "vector": _vec(3).tolist()})
    assert r.status_code == 200 and r.json()["stored"] == 1
    client.post("/api/ngram/flush")
    r = client.get("/api/ngram/lookup", params={"layer": 3, "token": 123})
    assert r.json()["found"] is True
