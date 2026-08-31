"""Weight Atlas analyzer: measured per-tensor quantizability as a prior.

Data: ~/atlas/public/models/qwen3.8-27b/atlas.jsonl (alesha-pro/atlas, MIT) —
measured INT8-per-channel / INT4-group128 / FP8-e4m3 SQNR, outlier-channel
ratios, and spectra for every weight tensor of the duty 27B checkpoint.

Uses here (see BICAMERAL_FRAMEWORK §6.2/§6.5):
- tap_candidates: rank full-attention layers as PLE/XYZ capture points
  (rich-but-clean hidden geometry: high stable_rank on the output-side
  mixing tensors, low outlier-channel load)
- int4_gate_table: the measured replacement for §6.5's synthetic INT4 floor
  (promotion still requires the live row_cos gate on real K rows)

The atlas knows nothing about activations — it is a prior, not a gate.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

DEFAULT_ATLAS = str(
    Path.home() / "atlas/public/models/qwen3.8-27b/atlas.jsonl"
)

# Output-side mixing tensors drive the hidden geometry at that layer.
_MIX_COMPONENTS = ("attn.o", "mlp.down")


def load(path: str | Path | None = None) -> list[dict]:
    p = Path(path or os.environ.get("ATLAS_JSONL", DEFAULT_ATLAS)).expanduser()
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _quant(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("sqnr_int4_g128") is not None]


def int4_gate_table(rows: list[dict]) -> list[dict]:
    """All measured 2-D tensors, worst INT4 first — the §6.5 measured floor."""
    return [
        {
            "name": r["name"], "layer": r.get("layer"), "component": r["component"],
            "sqnr_int4_g128": r["sqnr_int4_g128"], "sqnr_int8_ch": r["sqnr_int8_ch"],
            "sqnr_fp8_e4m3": r["sqnr_fp8_e4m3"],
            "row_amax_ratio": r.get("row_amax_ratio"),
            "stable_rank": r.get("stable_rank"),
        }
        for r in sorted(_quant(rows), key=lambda r: r["sqnr_int4_g128"])
    ]


def layer_report(rows: list[dict]) -> list[dict]:
    """Per-layer rollup across measured 2-D tensors, sorted by layer."""
    by_layer: dict[int, list[dict]] = {}
    for r in _quant(rows):
        if r.get("layer") is not None:
            by_layer.setdefault(r["layer"], []).append(r)
    out = []
    for layer, rs in sorted(by_layer.items()):
        ranks = [r["stable_rank"] for r in rs if r.get("stable_rank") is not None]
        ratios = [r["row_amax_ratio"] for r in rs if r.get("row_amax_ratio")]
        out.append({
            "layer": layer,
            "n_tensors": len(rs),
            "min_int4_db": min(r["sqnr_int4_g128"] for r in rs),
            "median_int4_db": statistics.median(r["sqnr_int4_g128"] for r in rs),
            "median_stable_rank": statistics.median(ranks) if ranks else None,
            "max_row_amax_ratio": max(ratios) if ratios else None,
        })
    return out


def tap_candidates(rows: list[dict], top: int = 8) -> list[dict]:
    """Rank layers as PLE/XYZ hidden-capture tap points.

    Heuristic (documented, not sacred): a good tap layer has high-rank
    output mixing (attn.o / mlp.down stable_rank — the hidden state carries
    rich structure) and low outlier-channel load (row_amax_ratio — cleaner
    geometry for the edge projection and any low-bit scoring downstream).
    """
    scored = []
    for rep in layer_report(rows):
        rs = [r for r in _quant(rows)
              if r.get("layer") == rep["layer"] and r["component"] in _MIX_COMPONENTS]
        if not rs:
            continue
        rank = statistics.median(r["stable_rank"] for r in rs
                                 if r.get("stable_rank") is not None)
        rep = dict(rep)
        rep["mix_stable_rank"] = rank
        rep["tap_score"] = rank - 2.0 * (rep["max_row_amax_ratio"] or 0.0)
        scored.append(rep)
    scored.sort(key=lambda r: -r["tap_score"])
    return scored[:top]


# ---------------------------------------------------------------------------
# Buffer residency map: atlas as the n-gram buffer's cached target map
#
# Each layer gets a residency tier and a recommended dct_topk compact rate,
# derived from measured weight-side signals:
#
#   hot  — tap candidates (rich hidden geometry worth capturing) or
#          INT4-fragile layers (min_int4_db below FRAGILE_DB: never quantize,
#          keep resident, prefetch first)
#   cold — robust layers (min_int4_db >= ROBUST_DB and low outlier load):
#          safe to swap out, quantize, or compact hard
#   warm — everything in between
#
# The map is a PRIOR from weight measurements. It sets defaults and prefetch
# order; the live row_cos gate still decides what actually ships (framework
# rule: no supersession without measured numbers).
# ---------------------------------------------------------------------------

# Data-driven quartiles of this checkpoint (min_int4_db across 64 layers:
# p25=15.55, p75=16.73; max_row_amax median=15.2). Tune per checkpoint.
FRAGILE_DB = 15.55
ROBUST_DB = 16.7
OUTLIER_RATIO = 15.0

_RATE_FOR_TIER = {"hot": 0.75, "warm": 0.5, "cold": 0.25}


def cache_map(rows: list[dict], tap_top: int = 8) -> dict:
    """Build the buffer residency map: {layer: {tier, rate, ...}}."""
    taps = {c["layer"] for c in tap_candidates(rows, top=tap_top)}
    layers: dict[int, dict] = {}
    for rep in layer_report(rows):
        layer = rep["layer"]
        fragile = rep["min_int4_db"] < FRAGILE_DB
        robust = (rep["min_int4_db"] >= ROBUST_DB
                  and (rep["max_row_amax_ratio"] or 0.0) < OUTLIER_RATIO)
        is_tap = layer in taps
        if is_tap or fragile:
            tier = "hot"
        elif robust:
            tier = "cold"
        else:
            tier = "warm"
        reasons = []
        if is_tap:
            reasons.append("tap-candidate")
        if fragile:
            reasons.append(f"int4-fragile({rep['min_int4_db']:.1f}dB)")
        if robust:
            reasons.append("robust")
        layers[layer] = {
            "tier": tier,
            "rate": _RATE_FOR_TIER[tier],
            "is_tap": is_tap,
            "min_int4_db": rep["min_int4_db"],
            "median_int4_db": rep["median_int4_db"],
            "median_stable_rank": rep["median_stable_rank"],
            "max_row_amax_ratio": rep["max_row_amax_ratio"],
            "reason": "+".join(reasons) or "default",
        }
    counts = {"hot": 0, "warm": 0, "cold": 0}
    for v in layers.values():
        counts[v["tier"]] += 1
    return {
        "model": "qwen3.8-27b",
        "tiers": counts,
        "prefetch_order": [l for l, v in sorted(
            layers.items(),
            key=lambda kv: ({"hot": 0, "warm": 1, "cold": 2}[kv[1]["tier"]],
                            -kv[1]["median_stable_rank"] or 0))],
        "layers": layers,
    }


def _entry(cmap: dict, layer: int) -> dict:
    layers = cmap["layers"]
    return layers[layer] if layer in layers else layers[str(layer)]


def tier_for(cmap: dict, layer: int) -> str:
    return _entry(cmap, layer)["tier"]


def rate_for(cmap: dict, layer: int) -> float:
    return _entry(cmap, layer)["rate"]
