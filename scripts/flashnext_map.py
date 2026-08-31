#!/usr/bin/env python3
"""Build a weight-loading map for Qwen3.8-Flash-Next-AWQ-INT4 without downloading weights.

Fetches only the safetensors headers (range requests) from the HF repo, then
aggregates per-tensor sizes into a loading plan: stationary core, expert pool,
PLE n-gram table, MTP/vision — with per-layer expert byte costs for sizing a
GPU-resident LRU expert cache.

Usage: python3 scripts/flashnext_map.py [--out data/flashnext_loadmap.json]
"""
import json
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
DTYPE_BYTES = {
    "I64": 8, "I32": 4, "I16": 2, "U8": 1, "U32": 4,
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
}
OUT = Path(__file__).resolve().parent.parent / "data" / "flashnext_loadmap.json"


def fetch_json(url, range_header=None):
    headers = {"Range": range_header} if range_header else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        return r.read()


def shard_header(shard):
    raw = fetch_json(f"{BASE}/{shard}", "bytes=0-7")
    n = struct.unpack("<Q", raw)[0]
    raw = fetch_json(f"{BASE}/{shard}", f"bytes=8-{8 + n - 1}")
    return json.loads(raw)


def tensor_bytes(entry):
    n = DTYPE_BYTES[entry["dtype"]]
    for d in entry["shape"]:
        n *= d
    return n


def classify(name):
    n = name
    if re.search(r"\.mlp\.experts\.\d+\.", n):
        return "expert"
    if "ngram_embed" in n or "ple" in n.lower():
        return "ple_ngram"
    if n.startswith("model.vision") or ".visual." in n:
        return "vision"
    if n.startswith("mtp.") or ".mtp." in n:
        return "mtp"
    if "embed_tokens" in n or "lm_head" in n:
        return "embed_head"
    return "core"


def main():
    idx = json.loads((Path("/tmp/flashnext_index.json")).read_text())
    shards = sorted(set(idx["weight_map"].values()))
    print(f"{len(shards)} shards, {len(idx['weight_map'])} tensors", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=8) as ex:
        headers = list(ex.map(shard_header, shards))

    # Aggregate
    by_class = {}
    by_dtype = {}
    expert = {}   # (layer, expert_id) -> bytes (weight+scale+shape rows for that expert)
    expert_aux = 0
    layer_re = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.")
    core_layers = {}

    for h in headers:
        for name, e in h.items():
            if name == "__metadata__":
                continue
            b = tensor_bytes(e)
            cls = classify(name)
            by_class[cls] = by_class.get(cls, 0) + b
            by_dtype[e["dtype"]] = by_dtype.get(e["dtype"], 0) + b
            m = layer_re.search(name)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                expert[key] = expert.get(key, 0) + b
            elif cls == "core" and ".layers." in name:
                lm = re.search(r"layers\.(\d+)\.", name)
                if lm:
                    l = int(lm.group(1))
                    core_layers[l] = core_layers.get(l, 0) + b

    GB = 1 << 30
    layers = sorted({l for l, _ in expert})
    n_experts = max(e for _, e in expert) + 1
    per_expert = [expert[k] for k in sorted(expert)]
    per_layer_expert = {l: sum(expert.get((l, e), 0) for e in range(n_experts))
                        for l in layers}

    plan = {
        "repo": REPO,
        "totals_gb": {k: round(v / GB, 2) for k, v in sorted(by_class.items())},
        "dtype_bytes_gb": {k: round(v / GB, 2) for k, v in sorted(by_dtype.items())},
        "experts": {
            "layers": len(layers),
            "experts_per_layer": n_experts,
            "bytes_per_expert_mb": round((sum(per_expert) / len(per_expert)) / (1 << 20), 2),
            "bytes_per_layer_gb": round(sum(per_layer_expert.values()) / len(layers) / GB, 3),
            "per_layer_expert_gb": {str(l): round(b / GB, 3) for l, b in sorted(per_layer_expert.items())},
            # top-10 routing, all layers, worst case no cache hits:
            "bytes_per_token_top10_gb": round(sum(per_layer_expert.values()) * 10 / n_experts / GB, 3),
        },
        "core_per_layer_mb": {str(l): round(b / (1 << 20), 1) for l, b in sorted(core_layers.items())},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k != "core_per_layer_mb"}, indent=2))
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
