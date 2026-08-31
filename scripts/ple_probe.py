#!/usr/bin/env python3
"""Phase 0 probe: is the Qwen3.8-Flash-Next PLE n-gram table q4-quantizable?

Samples random rows from several PLE table shards via HTTP range requests
(no bulk download) and measures SQNR for q4_0-style blocks (32-wide absmax),
INT4 group-128, and INT8 per-row — same metric family as neoharness/measure.py.

Gate: median q4_0 SQNR >= 25 dB -> PLE can be q4_0 (95GB -> 24GB).
"""
import json
import struct
import sys
import urllib.request

import numpy as np

REPO = "Qwen/Qwen3.8-Flash-Next"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
INDEX = "/tmp/fn_bf16_index.json"
ROWS_PER_SHARD = 256
SHARD_IDS = [0, 41, 83, 127]  # spread across the table
rng = np.random.default_rng(0xB1C4)


def fetch(url, range_header=None):
    req = urllib.request.Request(url, headers={"Range": range_header} if range_header else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def bf16_to_f32(buf):
    u = np.frombuffer(buf, dtype="<u2").astype("<u4") << 16
    return u.view("<f4")


def header_for(shard_file):
    raw = fetch(f"{BASE}/{shard_file}", "bytes=0-7")
    n = struct.unpack("<Q", raw)[0]
    return json.loads(fetch(f"{BASE}/{shard_file}", f"bytes=8-{8 + n - 1}")), 8 + n


def sqnr_db(w, wq):
    num = float(np.sum(w.astype(np.float64) ** 2))
    den = float(np.sum((w - wq).astype(np.float64) ** 2))
    return 10 * np.log10(num / max(den, 1e-30))


def quant_block(w, group, bits):
    """Symmetric absmax quant, per `group`-sized block along last axis."""
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    pad = (-cols) % group
    wp = np.pad(w, ((0, 0), (0, pad)))
    g = wp.reshape(rows, -1, group)
    amax = np.maximum(np.abs(g).max(axis=-1, keepdims=True), 1e-12)
    scale = amax / qmax
    q = np.round(g / scale).clip(-qmax - 1, qmax)
    dq = (q * scale).reshape(rows, -1)[:, :cols]
    return dq


def main():
    idx = json.load(open(INDEX))
    wm = idx["weight_map"]
    results = {"q4_0_b32": [], "int4_g128": [], "int8_row": []}
    norms, amaxes = [], []

    for sid in SHARD_IDS:
        tname = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{sid}.weight"
        sfile = wm[tname]
        h, data_start = header_for(sfile)
        e = h[tname]
        assert e["dtype"] == "BF16" and len(e["shape"]) == 2, e
        n_rows, n_cols = e["shape"]
        row_bytes = n_cols * 2
        t0 = data_start + e["data_offsets"][0]
        rows = rng.choice(n_rows, ROWS_PER_SHARD, replace=False)
        buf = bytearray()
        for r in sorted(rows):
            buf += fetch(f"{BASE}/{sfile}", f"bytes={t0 + r * row_bytes}-{t0 + (r + 1) * row_bytes - 1}")
        w = bf16_to_f32(bytes(buf)).reshape(len(rows), n_cols)
        norms.append(np.linalg.norm(w, axis=1))
        amaxes.append(np.abs(w).max(axis=1))
        for name, (group, bits) in {"q4_0_b32": (32, 4), "int4_g128": (128, 4), "int8_row": (n_cols, 8)}.items():
            results[name].append(np.array([sqnr_db(row[None, :], quant_block(row[None, :], group, bits)) for row in w]))
        print(f"shard {sid}: sampled {len(rows)}/{n_rows} rows x {n_cols}", file=sys.stderr)

    norms = np.concatenate(norms)
    amaxes = np.concatenate(amaxes)
    print(f"\nrows: {len(norms)}  row-norm p5/p50/p95: "
          f"{np.percentile(norms, 5):.2f}/{np.percentile(norms, 50):.2f}/{np.percentile(norms, 95):.2f}"
          f"  row-amax p50/p99: {np.percentile(amaxes, 50):.3f}/{np.percentile(amaxes, 99):.3f}")
    out = {}
    for name, arr in results.items():
        a = np.concatenate(arr)
        out[name] = {p: round(float(np.percentile(a, int(p[1:]))), 1) for p in ("p5", "p25", "p50", "p75", "p95")}
        print(f"{name:12s} SQNR dB  " + "  ".join(f"{k}={v}" for k, v in out[name].items()))
    gate = out["q4_0_b32"]["p50"] >= 25 and out["q4_0_b32"]["p5"] >= 15
    print(f"\nGATE (median>=25dB, p5>=15dB): {'PASS -> q4_0 viable' if gate else 'FAIL -> use q8_0'}")
    json.dump({"shards": SHARD_IDS, "rows_per_shard": ROWS_PER_SHARD, "sqnr": out, "gate_pass": gate},
              open("data/ple_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
