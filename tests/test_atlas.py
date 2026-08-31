"""Tests for atlas.py + measure.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neoharness import atlas, measure


def test_load_real_atlas():
    rows = atlas.load()  # ~/atlas checkout
    assert len(rows) == 1199
    assert any(r.get("sqnr_int4_g128") is not None for r in rows)


def test_int4_gate_table_sorted():
    table = atlas.int4_gate_table(atlas.load())
    assert table[0]["sqnr_int4_g128"] <= table[-1]["sqnr_int4_g128"]
    assert all("name" in t and "layer" in t for t in table)


def test_layer_report_covers_stack():
    report = atlas.layer_report(atlas.load())
    layers = [r["layer"] for r in report]
    assert layers == sorted(layers) and len(layers) == 64
    l63 = next(r for r in report if r["layer"] == 63)
    assert l63["n_tensors"] == 7


def test_tap_candidates_ranked():
    cands = atlas.tap_candidates(atlas.load())
    assert cands
    assert all(cands[i]["tap_score"] >= cands[i + 1]["tap_score"]
               for i in range(len(cands) - 1))


def test_cache_map_structure():
    cmap = atlas.cache_map(atlas.load())
    assert cmap["tiers"]["hot"] + cmap["tiers"]["warm"] + cmap["tiers"]["cold"] == 64
    assert set(cmap["prefetch_order"]) == set(range(64))
    for layer in range(64):
        tier = atlas.tier_for(cmap, layer)
        assert tier in ("hot", "warm", "cold")
        assert atlas.rate_for(cmap, layer) == {"hot": 0.75, "warm": 0.5, "cold": 0.25}[tier]


def test_cache_map_rules():
    cmap = atlas.cache_map(atlas.load())
    layers = cmap["layers"]
    # INT4-fragile GDN input layers must be hot (never quantize)
    assert layers[32]["tier"] == "hot" and "int4-fragile" in layers[32]["reason"]
    # tap candidates are hot
    assert all(layers[c["layer"]]["is_tap"] for c in atlas.tap_candidates(atlas.load()))
    # JSON roundtrip stringifies keys; lookups still work
    dumped = json.loads(json.dumps(cmap))
    assert atlas.tier_for(dumped, 32) == "hot"


def test_measure_gaussian_matrix():
    rng = np.random.default_rng(0)
    m = measure.measure(rng.standard_normal((64, 512)))
    # INT8 > FP8 > INT4 in quality, all in a sane dB band
    assert m["sqnr_int8_ch"] > m["sqnr_fp8_e4m3"] > m["sqnr_int4_g128"]
    assert 30 < m["sqnr_int8_ch"] < 60
    assert m["row_amax_ratio"] < 2.0  # gaussian rows are uniform


def test_measure_detects_outlier_row():
    w = np.random.default_rng(1).standard_normal((32, 256)) * 0.01
    w[7] *= 100  # one hot row
    m = measure.measure(w)
    assert m["row_amax_ratio"] > 10
    # per-row/per-group scales adapt, so self-SQNR stays high — the ratio
    # (not SQNR) is the matmul-interaction risk signal
    assert m["sqnr_int8_ch"] > 30


def test_measure_rejects_1d():
    with pytest.raises(ValueError):
        measure.measure(np.zeros(10))


def test_fp8_grid_properties():
    g = measure._E4M3
    assert g[0] == 0.0 and g[-1] == 448.0
    assert measure.quant_fp8_e4m3(np.array([[1.0, -448.0, 1000.0]])).max() <= 448.0
