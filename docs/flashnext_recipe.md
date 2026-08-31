# Qwen3.8-Flash-Next on 2× Arc Pro B70 — Conversion & Launch Recipe

Target: replace the duty 27B with Qwen3.8-Flash-Next as the NeoHarness backend.
Hardware: this box — 2× Intel Arc Pro B70 (32GB VRAM each, PCIe, no XeLink),
31GB host RAM, ~400GB free NVMe. Runtime: llama.cpp (PR ggml-org#27742,
`qwen4_exp` arch) with ESIMD MoE kernels in the ggml SYCL backend path.

vLLM is **not** an option: the official recipe states XPU is unsupported and
PLE CPU offload is NVIDIA-only.

## Memory budget (post-quantization)

| Section | Source size | Our quant | Bytes | Home |
|---|---|---|---|---|
| Core (attn/GDN/shared-expert/norms) | 6.8GB BF16 | q8_0 | ~7GB | GPU0 |
| embed_tokens + lm_head | 2.4GB BF16 | q8_0 | ~2.5GB | GPU0 |
| MTP head | 4.9GB BF16 | q4_K_M | ~2.8GB | GPU0 |
| Expert pool (512×48, ~113B params) | ~120GB BF16 | q4_K_M | ~65GB | split |
| — hot experts | | | ~40GB | GPU0+GPU1 VRAM (~60% of pool) |
| — cold experts | | | ~25GB | mmap'd NVMe (not RAM — see below) |
| PLE n-gram table | 95.4GB BF16 | q6_K (Phase 0) | ~40GB | mmap'd NVMe, host-side gather |
| KV + GDN state (32K ctx) | | | ~4GB | GPU0 |

Hard constraint: host RAM is 31GB, so PLE + cold experts (~49GB) **cannot** be
RAM-resident. Both live mmap'd from NVMe (`--no-mmap` off, page cache does the
tiering; keep RAM headroom for the OS + fabric + harness). Expert/PLE misses
then cost an NVMe read (~7GB/s gen4) instead of a PCIe transfer — this is why
VRAM expert residency (60% + routing locality) is the throughput lever.

Disk choreography: BF16 source is ~335GB and the GGUF output ~104GB, which
exceeds free space together. Convert shard-streaming and delete BF16 shards
as they are consumed (convert_hf_to_gguf reads sequentially per shard).

## Phase 0 — validate PLE quantizability — DONE (2026-08-30, gate resolved)

Measured via `scripts/ple_probe.py` (range-request row sampling, no bulk
download; 1024 rows across shards 0/41/83/127; results in `data/ple_probe.json`).
Table geometry: 128 shards × [2.5M rows, 160]. Rows are extremely uniform
(norm p5–p95: 0.08–0.11; amax p99 0.032) — no outlier problem.

SQNR ladder (block-32 absmax, dB, p5/median/p95):

| quant | p5 | p50 | p95 | PLE size |
|---|---|---|---|---|
| q4 | 19.3 | 20.4 | 21.4 | ~24GB |
| q5 | 25.8 | 27.0 | 28.1 | ~33GB |
| q6 | 32.1 | 33.3 | 34.3 | ~40GB |
| q8 | 44.4 | 45.6 | 46.6 | ~48GB |

Decision: **q6_K for the PLE table** (~40GB mmap'd NVMe). q4's 20dB fails the
25dB gate; q8's 45dB is overkill for a gather table whose output passes through
learned key/value projections. q6 at 33dB is ~0.05% noise energy — safely
inaudible downstream. If Phase 2 bench shows PLE-linked quality loss, escalate
to q8_0; if RAM/disk pressure dominates, q5_K at 27dB is the floor.

## Phase 1 — convert

```bash
# source: official BF16 repo, NOT the cyankiwi AWQ build
# (converter expects plain BF16 safetensors; compressed-tensors input is unsupported)
hf download Qwen/Qwen3.8-Flash-Next --local-dir "$MODEL_DIR/flashnext-bf16"

python convert_hf_to_gguf.py "$MODEL_DIR/flashnext-bf16" \
  --outfile "$MODEL_DIR/flashnext-bf16.gguf" --outtype bf16
# delete BF16 shards as they are consumed to stay under 400GB

./llama-quantize \
  --override-tensor "ngram_embed=q6_K" \
  --override-tensor "\.ffn_(up|gate|down)_exps\.weight=q4_K_M" \
  "$MODEL_DIR/flashnext-bf16.gguf" \
  "$MODEL_DIR/flashnext-q4km.gguf" q8_0
# PLE = q6_K per Phase 0 measurement. Base q8_0 keeps core/embeddings wide.
# (adjust the ngram_embed regex to the actual GGUF tensor names after conversion)
```

## Phase 2 — launch (PROVEN 2026-08-31, do not regress)

Build: `~/llama.cpp-flashnext/build-sycl-hs` (has the local /v1/hidden_states
patch + n_embd_out_impl fix; uncommitted working tree). Env: oneAPI setvars,
ONEAPI_DEVICE_SELECTOR=level_zero:*.

```bash
./llama-server -m "$MODEL_DIR/flashnext-q4km-pleq6k.gguf" \
  --device SYCL0,SYCL1 --tensor-split 1,1 \
  -ngl 99 \
  -ot "per_layer_token_embd=CPU,blk\.[0-9]*[048]\.ffn_.*_exps=CPU" \
  -c 49152 -fa on -t 14 -np 1 \
  --embeddings -b 1536 --ubatch-size 1536 \
  --host 127.0.0.1 --port 8001
```

Measured: pp ~50 t/s, tg ~20 t/s warm. Hard-won lessons:

- `-ot` is comma-separated; repeated flags drop all but the last.
- Never use `--n-cpu-moe` with multi-GPU: it offloads the FIRST N layers,
  which tensor-split then piles onto one device -> OOM. Interleave CPU layers
  with a regex (endings 0/4/8 = 15 of 48 layers on CPU) instead.
- 10 CPU layers OOMs on load; 15 leaves ~3GB/device headroom.
- No `--warmup`: it pre-reserves the worst-case compute pool and leaves GPU0
  at ~42MB free; the first real request then OOMs the SYCL memory pool.
- `--embeddings` sizes a logits buffer of n_vocab x n_batch; -b 2048 OOMs,
  1536 fits and covers fabric's 2048-token capture cap.
- vllm.service (27B) must stay disabled; it auto-starts on login and will
  fight llama-server for GPU0 and page cache (observed: tg collapse to 1.7).

MTP: NOT exported (converter sets no_mtp=True for qwen4exp); wiring it means
converter + nextn graph work — see "Open risks".

ESIMD kernels slot in as the ggml SYCL-backend MoE gemm implementation.
Acceptance gate: tg >= 8 t/s (currently ~20).
tg ≥ 8 tok/s decode.

## Phase 3 — harness swap

1. Point fabric/NeoHarness at `:8001` (chat completions only — the
   `/v1/completions` MTP landmine presumably still applies).
2. Hidden geometry changes: hidden 2560, 48 layers, full-attn layers are
   3,7,11,...,47 (every 4th). Set `NEOH_TAP_LAYER=47` (or measure; GDN layers
   may gate poorly — prefer a full-attn tap).
3. PLE worker affinity (`ZE_AFFINITY_MASK=1`) now conflicts with the inference
   GPU — re-point the NeoHarness XYZ worker to GPU0 or accept sharing.
4. Sampling: model card's agentic settings match ours (temp 1.0, top_p 0.95);
   keep `preserve_thinking` on — it is the model-side analogue of our BPE
   persistence and the card says it improves agentic consistency.
5. Run `scripts/bench_longhorizon.py` multi-seed against the new backend;
   promotion gate: BPE ≥ raw on the ledger task, tg ≥ 8 tok/s.

## Open risks

- llama.cpp PLE gather is host-side (per PR description) — confirm the table
  never demands VRAM, else budget breaks.
- MTP speculative support in llama.cpp for this arch is unconfirmed.
- NVMe-mmap expert misses cap decode; if tg < 5 tok/s, the fallback is
  quantizing experts to q3_K_M (~50GB, ~75% VRAM residency).
- QSA indexer landed in the PR "not wired up yet" at first — confirm the
  merged state includes it; without QSA, long-context retrieval quality drops.
