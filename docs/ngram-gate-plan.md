# Plan: wiring the rotary sliding gate to the ngram store

Status: PLAN DRAFTED, no code changes yet for this phase.
Author context: continues "Retry MRL Embeddings and Reranker on *.107".
Box: 192.168.1.107 (fish default shell; use `bash -s` heredocs for non-trivial commands).

Purpose of this file: survive a dead session. It is the source of truth for the
ngram work. Read it before touching code.

---

## 0. Verified facts (do not re-derive; re-verify only if a number is needed)

- Strata is a llama.cpp FORK. `~/Strata/third_party/{llama.cpp,ggml}` are git
  submodules pointing at Strata's OWN repo (Niko1221/Strata, commit 7695859),
  not upstream ggml-org. So "fork Strata again" is the correct move.
- The main model on :8080 is Strata IQ1_M (`strata-coder-iq1_m.json`),
  speculative decoding = MTP heads from `~/Strata-data/mtp/rt` (`--spec 4
  --spec-min-p 0.5`).
- The ngram capability already EXISTS in Strata as `--lookup-chain`
  (`sycl/src/program/generate.cpp:682`; `--lookup-chain K` clamp 0..7, default 0
  = OFF; `--lookup-chain-min M` default 3). It is OFF in the live config.
- `--lookup-chain` is the PIPELINED speculative decoding from llama.cpp issue
  #23184 (draft-mtp -> ngram chained). Already implemented in Strata. See §2.
- `DraftSource` (`include/strata/spec/draft_source.hpp`) is the extension point.
  Extra sources register via `extra_draft_sources()`; engine keeps the proposal
  with the longest match. Registration comment explicitly names "a retrieval
  store over a codebase or earlier conversations" as a target — that is the
  harness LTM.
- `SuffixDrafter` (`include/strata/spec/suffix_drafter.hpp`) is the ngram store:
  trigram-keyed, WAYS=4 open-addressing, min_match=3, max_match=32,
  capacity_tokens=1<<19, ~20 bytes/history token, O(1) appends, PER-REQUEST
  (reset() on new request). No shared pool, no 0.25 occupancy cliff.
- Embedder: Qwen3-VL-Embedding-2B on :8099, wired in `neoharness/ltm.py`
  (EMBED_BASE) but OFF by default (`NEOH_RECALL_EMBED=0`); all runs so far used
  `LTMStore(embed=False)`. Recall has been lexical-only.
- Reranker: Qwen3-VL-Reranker-2B on :8097 is RUNNING but has ZERO references in
  any harness .py. Never wired.
- Model-side ngram (separate thing): `ngram_embed` / PLE n-gram embedding lives
  in the Qwen3.8-Flash-Next weights, consumed in the model graph
  (`qwen4exp.cpp`, `Strata/sycl/src/ngram/ple_reader.cpp`, `--ple-gguf`). The
  recipe quantizes `ngram_embed=q6_K`. This is NOT the speculative ngram.

Earlier session's sliding-gate work (context, already landed):
- Fixed `SlidingGate.apply()` positive-feedback bug (sine scale was the measured
  window, so nothing was ever retired). Now uses configured `stm_slots`.
- `MSGS_PER_SLOT` default raised 1.0 -> 2.0.
- Verified: 140/140 pytest pass; secret retires to LTM and returns via recall.

---

## 1. Objective and the gate-semantics question

Objective: increase throughput/quality on long-horizon agentic runs by letting
the sliding gate (STM/LTM sine, XYZ retirement) inform the ngram drafting path.

Open question that blocks gate design — the drafter keeps its OWN full `hist_`
of every request token, independent of the sliding gate's window decisions.
When the gate retires a span to LTM, the SuffixDrafter still holds those tokens.
So "what does the gate do to the ngram store?" has at least three candidate
semantics, to be chosen AFTER measurement:

  S1. Bound the drafter's history to the live window (mirror the gate's cut).
  S2. Consult an LTM-backed DraftSource (retrieval over retired spans) — the
      "retrieval store over earlier conversations" the header names.
  S3. Modulate match depth (min_match/max_match) over the sine.

None of these is chosen yet. Measurement first (Phase 2) decides.

---

## 2. Relationship to llama.cpp #23184 (already solved here)

The linked issue wants draft-mtp and ngram chained so ngram EXTENDS mtp drafts
instead of running an independent stream. Strata's `--lookup-chain` does exactly
this: after the MTP drafts, the lookup is invoked on a context tail that ENDS
WITH the unverified MTP drafts (`n_pending`), and the lookup proposal is
appended to the same verify window. So there is no independent-stream problem
here and no upstream fix is needed. `--spec-type draft-mtp,ngram-mod` (the
llama.cpp spelling) does not exist in Strata; the equivalent flag is
`--lookup-chain`.

llama.cpp `ngram-mod` for reference (NOT used here): 4M-entry direct-mapped LCG
hash pool (n_match=24), SHARED across server slots, 0.25-occupancy reset cliff.
Strata's SuffixDrafter differs on all three points (trigram-keyed, per-request,
no cliff) — so the "0.25 reset cliff" is a non-issue and is compatible with the
quaternion rotation by construction.

---

## 3. Phased plan (each phase has entry + exit criteria)

### Phase 1 — Rebuild sanity (entry: now)
- Rebuild Strata from the existing working tree (build-sycl-aot, SYCL/oneAPI).
- Exit: `strata --serve` starts and serves a short completion on a TEST port.
- Risk: low (box built it before), but do this BEFORE code changes.

### Phase 2 — Measure `--lookup-chain` as-is (no gate yet)
- Copy `strata-coder-iq1_m.json` to a variant; add `--lookup-chain 7`
  (and consider `--lookup-chain-min 3`). Leave the live config untouched.
- Benchmark acceptance-rate delta vs MTP-only on agentic-shaped prompts:
  repeated tool calls, JSON, refactor/quoted text, reasoning-repeated-in-answer.
- Instrument: SuffixDrafter match length + chain accept counts; MTP accept rate.
- Exit: a NUMBER for acceptance-rate delta. If it does not move, there is no
  problem to gate — stop and reconsider.
- This is the "measure, don't tune what isn't broken" rule.

### Phase 3 — Choose gate semantics (S1/S2/S3)
- Use Phase 2 data + the drafter-vs-gate history divergence to pick one.
- Decide whether the gate writes into `SuffixDrafter::table_`, or registers a
  new `DraftSource`, or both.

### Phase 4 — Implement + gate regression tests
- Fork point: new branch in `~/Strata` working tree (already a fork of itself).
- Add law tests mirroring the sliding-gate ones (system head never retired;
  window bounded; gate does not break when DraftSource returns 0).

### Phase 5 — Future: embedder + reranker (NOT this phase)
- Embedder is already wired, just off: set `NEOH_RECALL_EMBED=1` (MRL 256-dim).
- Reranker :8097 needs a NEW client in the harness (never written).
- These change gate semantics: the reranker is a selection function over LTM
  entries, so it would govern WHAT IS WORTH RECOVERING, a different axis from
  the ngram drafter's WHAT IS WORTH DRAFTING. Two coupled decisions.

---

## 4. Method rules carried from the earlier session

- Measure before tuning. Print a number, not a plausible window.
- Re-verify preconditions in the TEST, not by reading source. (The 0.25 cliff and
  "no ngram path" were both wrong conclusions from grepping upstream identifiers
  instead of searching for the capability.)
- An unreachable optional dependency must never silently degrade a turn
  (precedent: embedder is opt-in by design for this reason).

---

## 5. Cold-resume checklist

1. Read this file.
2. Reconnect: `ssh bbear@192.168.1.107` (SSHPASS=0666, fish shell).
3. `cd ~/flashnext-harness && python3 -m pytest tests/ -q`  # expect 140 passed.
4. Check live config for `--lookup-chain` (currently absent).
5. Continue at the first phase whose exit criteria are unmet.

---

## 6. Phase 2 RESULT (measured 2026-10-12, box .107)

Status: Phase 2 EXIT CRITERION MET. There IS something to gate. Phase 3 may start.

### Method (all in `~/phase2-bench/`, scratch dir, repo untouched)
Three arms, each a FRESH server process (no prompt-cache carryover), run SEQUENTIALLY
because only one engine fits in VRAM. Arms differ ONLY in the drafting flags; all other
live flags are byte-identical (vision dropped: separate mmproj engine, no text-path effect).

  C-mtponly   --suffix-draft 0                     <- true MTP-only reference
  A-baseline  (live default)                       <- what runs today
  B-chain7    + --lookup-chain 7 --lookup-chain-min 3

n = 24 distinct prompts per arm (12 verbatim-repetition, 12 non-repetitive controls),
greedy/temperature 0, `reasoning_budget_tokens=32`, `max_tokens=512`.
NO code changes were needed: the engine already emits per-request
`drafts accepted X of Y` plus `suffix drafts:` / `lookup chain:` counters on stderr,
and the server returns `timings.draft_n` / `draft_n_accepted` per request.

### Two corrections to the plan's premises
1. "MTP-only baseline" was wrong. `--suffix-draft` DEFAULTS TO 3 and the live config
   passes neither flag, so prompt-lookup drafting is ALREADY ON in production today.
   Arm A is "live default", not "MTP only". Arm C is the true MTP-only reference.
2. "Acceptance-rate delta" is the wrong headline metric. Acceptance rate FELL while
   throughput ROSE - see below. §1's real objective is throughput.

### THE NUMBERS (mean decode tok/s, API-side, n=24)
  Arm                MTP acceptance   mean tok/s
  C-mtponly              79.27%          81.67
  A-baseline (live)      78.34%          81.86
  B-chain7               75.91%          84.80

  Acceptance-rate delta B vs A:  -2.43 pp   (DOWN)
  Throughput   delta B vs A:  +2.94 tok/s  (+3.6% mean)

### The effect is entirely conditional - this is the real finding
  VERBATIM-repetition prompts (n=12): +6.28 tok/s, 95% CI [+3.81, +8.76], 11/12 win
  CONTROL, non-repetitive    (n=12): -0.39 tok/s, 95% CI [-0.82, +0.03],  2/12 win
  (CI includes 0 => no effect on non-repetitive work)

  Mechanism: the chain fires in only 114/3365 decode windows (3.4%), accepts 199/326
  chained drafts (61.0%), contributing 1.98% of generated tokens. It also shifts work
  away from the suffix drafter (suffix offered 1444 -> 2564, acceptance 86.1% -> 82.5%).

### What this means for Phase 3
The win is real but CONDITIONAL: it pays only where the answer repeats spans verbatim.
The open question for S1/S2/S3 is therefore no longer "is there a problem" but
"how do we switch the chain ON for repetitive work without paying the acceptance-rate
cost on non-repetitive work". S3 (modulate match depth over the sine) is the axis
these numbers point at. Nothing is implemented yet; no Strata code has been touched.

### Caveats
- Output capped at 512 tok/prompt. Long-horizon behaviour at 10k+ tokens is UNMEASURED.
- `--lookup-chain 7` only; K=1..7 sweep not done. Window cap is 8.
- `--vision` excluded from all arms (live keeps it).

---

## 7. Phase 3 CLOSED (2026-10-12): S3 rejected by measurement; no new gate is needed

### S1/S2/S3 decision
**None of S1/S2/S3 should be built as a gate.** The gating already exists inside the
engine and beats anything the sliding gate could supply:

`DraftPolicy` (include/strata/spec/draft_policy.hpp) decides per verify round whether the
chained lookup pays, against MEASURED round costs and conditioned on MATCH LENGTH - i.e.
it learns the "is this text repetitive" discrimination online. That is exactly the signal
Phase 2 found predicts the win. The sine lives in the Python harness, the drafter in the
C++ engine; there is no per-request channel, and the knob it would drive is already optimal.

### Depth sweep - the knob S3 wanted to modulate is already at its floor
All arms `--lookup-chain 7`, n=24 prompts, paired per-prompt 95% CI:

  depth 3 (default floor) vs chain OFF : verbatim +9.84 tok/s [+6.78,+12.90] 12/12  SIGNIFICANT
  depth 3 vs 6                          : verbatim -7.58 tok/s [-10.66,-4.49] 1/12
  depth 3 vs 12                         : verbatim -7.67 tok/s [-10.86,-4.47] 0/12
  depth 24                              : chain never fires (0/3 drafts, 1/3535 windows)

Raising `min_match` monotonically destroys the effect. min 3 is the clamp floor and is
the clear winner for mean AND verbatim tok/s. The useful signal lives at the SHALLOWEST depth.

### DraftPolicy earns its keep
  policy ON vs OFF (STRATA_LOOKUP_CHAIN_FIXED=1): verbatim +4.14, control +3.95 tok/s better ON.
  Forced chain: acceptance collapses 63.8% -> 31.3%, MTP acceptance 77.0% -> 67.7%,
  chain fires 13.4% of windows instead of 4.6%.

### The payoff needs NO condition
Control (non-repetitive) arm is +0.95 tok/s [+0.19,+1.71] - positive, CI excludes zero.
So no repetitive-prompt gate is needed: DraftPolicy already suppresses the chain where it
does not pay. The entire deliverable of Phases 2-3 is: add `--lookup-chain 7` to the live config.

### The vendor ngram pipeline: mapped, not missing
raw->context->index->table->offload->host->prefetch->vram->active emb->compute is the PLE
path and is ALREADY implemented: --ple-io direct|mmap|ram (default direct, unbuffered SSD,
table never enters RAM/file cache), --ple-inflight 256, prefetch_rows() async prefetch
during GPU work, --ple-row-cache 1<<20 rows (~95 MB), 16 rows/token, STRATA_PLE_TRACE=1.
Live: read probe 117579 rows/s (fine); ZERO 'waiting for the drive'/OVERFLOW events.
Shard 2 IS the table (320,001,536 x 90 B = 28,800,138,240 vs file 28,800,138,432; the
192 B delta is the GGUF header) => it is a WEIGHT table and carries NO conversational context.

### DEFERRED to Phase 5 (cross-session context)
The open question - can a project's context be held BETWEEN SESSIONS, or the window extended -
is NOT the speculative drafter's default and is NOT Phase 3's job. Relevant facts:
  - the right artifact is SuffixDrafter, not PLE; it is per-request today (`sfx.reset()`
    at generate.cpp:10373), sized `max_context + 4096`;
  - `DraftPolicy policy(S)` beside it is deliberately PROCESS-lifetime ("learned over the
    whole process"), so the asymmetry is one line, not an architectural change;
  - the extension point already exists: `extra_draft_sources()`;
  - making it persist IS S2 (LTM-backed DraftSource, "retrieval over retired spans").
  CAVEAT: a drafter is a SPEED mechanism with no attention state. It cannot free KV cells,
  and it only pays on EXACT repeats (measured: 4.6% of windows). Semantic recall needs the
  embedder+reranker axis (Phase 5), which is a different axis from the ngram drafter.

### Phase 4 reinterpreted
Phase 4's "implement" is empty (no gate code to write). Its value is now the regression
test that LOCKS IN the measured behaviour so a future change cannot silently regress
lookup-chain: depth stays at the floor 3, policy stays ON, and the verbatim-family win
(~+9.8 tok/s) survives.

---

## 8. Phase 5 INVESTIGATED (2026-10-12): embedder as a PARALLEL path; reranker not usable as served

Harness untouched by this phase (`neoharness/ltm.py` NOT edited). Benchmarks in `~/phase5-bench/`.
Corpus: 20 retired spans, 14 labelled queries, split SEMANTIC (10, low lexical overlap with the
gold span) / LEXICAL (4, shares query vocabulary). Measured against the REAL LTMStore.

### 8.1 The vision-tower theory: REFUTED on routing
Neither :8099 nor :8097 loads an mmproj (0 mmproj fds, 0 cmdline hits). :8099 returns 2048-dim
L2-normalised vectors (norm 1.0). Text goes through the TEXT path; no vision tower is engaged.
(Whether the VL VARIANT's text tower is as good as a text-only embedder is a separate, empirical
question - answered by the numbers below, not by architecture.)

### 8.2 MRL 256 is SAFE (validates the shipped EMBED_DIMS)
Embedding-only ranking vs full 2048-d, by truncation depth:
    64: 64% top3-agreement | 128: 71% | 256: 86% | 512: 76% | 1024: 90% | 2048: 100%
  recall@3 is IDENTICAL (94%) at 256, 1024 and 2048. 256 is a real Matryoshka prefix.
  The 512 dip is n=14 noise, not a property - do not read anything into it.

### 8.3 THE HEADLINE: the shipped embedding path gains NOTHING, by construction
  recall@3 vs 20 spans / 14 queries
    lexical only (production today)     SEMANTIC 4/10  LEXICAL 4/4  ALL  8/14 (57.1%)
    lexical + embed RE-RANK (shipped)   SEMANTIC 4/10  LEXICAL 4/4  ALL  8/14 (57.1%)  <- zero gain
    lexical UNION embed (proposed)      SEMANTIC 9/10  LEXICAL 4/4  ALL 13/14 (92.9%)
  Why: recall() takes a LEXICAL shortlist FIRST and returns [] when it is empty, so 5/14
  queries never reach the embedder at all (embed_calls=9). Every query that DOES reach it is
  one lexical already handles. The embedder is switched off exactly where it wins.

### 8.4 The reranker is NOT usable as served
  - pooled-embedding scoring (cos/distance against query or document): 0/14 top-1, BELOW the
    5% random baseline. A cross-encoder's relevance lives in a classification head over a
    yes/no logit, not in the pooled state -- `--pooling last` returns the pre-head vector.
  - the logit path IS reachable (/v1/completions logprobs); template A separates one triple
    correctly, but over the full corpus reranking the union top-8 leaves recall@3 UNCHANGED
    at 13/14 (92.9% -> 92.9%): it fixed one query and broke another. NEUTRAL.
  => Do not write the :8097 client yet. It costs 112 calls/14 queries (24s) for no gain.

### 8.5 What "context extension" actually needs, and its cost
The union path IS context extension, measured: 57.1% -> 92.9% recall@3, semantic family
4/10 -> 9/10. But it breaks the cost assumption in ltm.py's own docstring ("~130-190 ms per
call ... which is exactly why it only ever re-ranks a shortlist"): a union must embed every
entry it has not yet seen, bounded by MAX_ENTRIES=512. At the measured ~0.5 s/call that is a
first-pass cost of up to ~4 min. Any implementation MUST embed lazily and amortise (a small
per-recall budget, or embed on retire) - it must never embed the whole store inside one turn.

### CAVEATS
n=14 queries / 20 spans / synthetic corpus. These are directional, not production numbers.
Validate on real retired spans from actual runs before trusting 92.9%.

---

## 9. Phase 5 IMPLEMENTED: lexical-empty fallback (the union, in its measured form)

`neoharness/ltm.py` changed. `NEOH_RECALL_EMBED=1` is still required; the default is unchanged.

### What was added
1. `EMBED_SCAN_BUDGET` (default 48): max NEW entries one recall() may embed.
2. `_embed_pending(limit)`: lazy, incremental, and it STOPS on the first unreachable
   server - a dead :8099 costs one timeout, not one per entry.
3. In `recall()`, when the LEXICAL SHORTLIST IS EMPTY, the embedder is consulted over
   the store and its hits become the shortlist. Ordering stays lexical-first elsewhere,
   so the fallback can only ADD candidates that were previously invisible.
4. `stats()["embed_fallbacks"]` - the rescue is counted, never silent (same law as
   embed_failures).
5. recall() no longer spends a second timeout probing the query when the scan already
   proved the server is down.

### Measured, end to end against the LIVE :8099 (20 spans, 14 labelled queries)
    lexical only (production)        SEMANTIC 4/10  LEXICAL 4/4  ALL 8/14 (57.1%)   0.0 s
    lexical + embed FALLBACK (new)   SEMANTIC 8/10  LEXICAL 4/4  ALL 12/14 (85.7%)  7.6 s
  +28.6 pp overall; the semantic family doubles, 40% -> 80%. The fallback fired 5x,
  19 embeds, 0 failures. Cost is bounded: it pays only on a lexical-empty query.

### CORRECTION to §8.3
§8.3 reported 13/14 (92.9%) for the union. That number was WRONG: retire() dedups on
the TERM SET, so 20 spans collapse to 19 entries, and the harness matched the gold by
SPANS INDEX while recall returns ENTRY ids - off by one for every span after the
deduped one. Matching gold by TEXT gives the honest figure: 12/14 (85.7%).
The direction and the conclusion are unchanged; the size of the win was overstated.

### Still missed (2/14), both genuinely hard
  - "how do we get a change all the way to users in production safely?" (gold: canary ramp)
  - "what happens if a call fails partway through?" (gold: retries idempotent calls)
Neither is a term-overlap case nor an obvious paraphrase; this is where a reranker
would have to earn its place, and §8.4 showed the one on :8097 does not.

### Tests (151 pass; the 6 new ones were mutation-tested, all 5 mutants caught)
  lexical-empty query is rescued by the embedder | the fallback does not fire on a
  lexical hit | one recall embeds at most the budget | a dead embedder stops after ONE
  call | a dead embedder still yields the lexical answer | stats expose the fallback.
  NOTE for whoever edits next: retire() dedups on the term set, so a test that retires
  "entry 0".."entry 199" holds ONE entry and asserts nothing. Two of these tests were
  vacuous that way and were rewritten with distinct spans.

### NOT DONE: the reranker client. §8.4: neutral on recall, pooled scoring below random.

---

## 10. Compactor model sizing + the "rotary gate" question (2026-10-12)

### 10.1 VRAM: an 0.8B model DOES fit alongside the main engine
Qwen3.5-0.8B-Q6_K.gguf (609 MiB, 24 layers, 2 KV heads, head_dim 256, trained ctx 262144),
measured on card 1 with q8_0 KV, alongside the live engine (9.3 GiB free card0 / 10.2 GiB card1):

    -c 32768   -> 1.09 GiB
    -c 131072  -> 1.97 GiB
    -c 262144 (native) -> 3.14 GiB      <- fits, ~7 GiB headroom

So a native-262k compactor co-resident with the main engine is affordable. The binding
constraint is NOT parameter count; it is KV bytes/token. Do NOT confuse them.

### 10.2 BUT the 0.8B cannot actually COMPRESS (measured)
Prompts designed for extraction, 15 load-bearing facts (names/numbers/dates/flags), input
1589 chars, q8_0 KV, reasoning_budget=32 (the model is a THINKER - without a budget it burns
2500 tokens on reasoning and emits ZERO answer; same trap as Phase 2 pass 1):

    A bullets (original)         out 1955 chars  ratio 0.81x  facts 15/15  finish=length
    B hard length cap (120w)     out 1909 chars  ratio 0.83x  facts 14/15  finish=length
    C dense fact list            out 1858 chars  ratio 0.86x  facts 14/15  finish=length
    D distractor-free preamble   out 2046 chars  ratio 0.78x  facts 11/15  finish=length

    Every arm hit the 600-token cap (finish=length). It is not fact-loss - it CANNOT STOP.
    Ratio < 1.0 means it EXPANDS (it echoed my instructions as bullet 1). The user's thesis
    ("fewer parameters = better at restating in fewer words") does NOT hold at 0.8B.

### 10.3 The "rotary gate" / "quaternion rotation" does not exist in this codebase
VERIFIED by exhaustive search: `quaternion` appears ONLY in third-party miniaudio, numpy's
docstring, and THIS PLAN DOC's own line 87. Zero hits in Strata's working tree, zero across
all git refs/commits. What actually exists:
    - kv_q4.hpp        : Walsh-Hadamard rotation for Q4_0 KV (a QUANTIZATION transform)
    - rope_scaling     : RoPE rotation (position encoding)
    - expert_source.cpp: exchange-BUFFER rotation (memory ring)
    - verify.cpp       : slot rotation (batch scheduling)
None of these is a GATE; they are quantization / positional / scheduling mechanics.
Also note flash-next runs --kv int8, not Q4_0, so the Hadamard path is OFF live.

### 10.4 The paper (arXiv 2608.03893) is applicable in principle; the compactor does NOT need it
Cross-Model KV Transfer maps one model's KV cache into another's head-space so the receiver
SKIPS PREFILL. Its precondition = MATCHED-KV (same n_kv, same head_dim, same tokenizer).
BOTH models on this box satisfy that literally:
    flash-next: tokenizer.ggml.pre=qwen35, vocab 248320, n_kv=2, head_dim=256
    0.8B      : tokenizer.ggml.pre=qwen35, vocab 248320, n_kv=2, head_dim=256
BUT the paper's own success condition is not "matched-KV" alone - it is matched-KV *within a
family*. The paper's two FAILING pairs (Ministral 3B->14B, 8B->14B) ARE matched-KV and still
collapse to 42-44% retention (floor-norm 11-15%). Our pair is across architectures
(qwen4exp vs qwen35) AND 48 vs 24 layers - the large depth gap is exactly the regime the
paper shows failing. Treat this as NOT validated for this pair.

WHY THE COMPACTOR DOESN'T NEED IT: KV transfer exists to skip PREFILL of the SAME tokens.
A compactor REWRITES tokens (restates them), so the output is different text - there is no
source KV cache to donate. A 0.8B prefill is already ~0ms of the cost the paper saves.
The real blocker remains 10.2 (it cannot stop), not prefill.

---

## 11. Compaction compared: 0.8B vs 4B vs no model at all (2026-10-12)

Same 1589-char retired-span input, same 15 load-bearing facts, greedy, reasoning capped.
Ratio = input/output chars. Facts = load-bearing facts surviving verbatim.

  approach                       ratio    facts   finish
  Qwen3.5-0.8B  (4 prompts)     0.78-0.86x 11-15/15  ALL length  <- EXPANDS, never stops
  Qwen3.5-4B    compressed block  2.20x   13/15   stop         <- the only arm that terminated
  Qwen3.5-4B    tight cap 120w   2.19x   13/15   length
  Qwen3.5-4B    telephone        2.87x   11/15   length
  4B @ max_tokens=150 (all 3)   3.4-4.5x  6-9/15   length      <- ratio bought with facts
  clause extractive @50% budget  1.97x   14/15   n/a (deterministic, no model, no VRAM)
  clause extractive @40%         2.44x   11/15   n/a
  clause extractive @30%         3.28x   10/15   n/a
  clause extractive @25%         4.00x    7/15   n/a
  clause extractive @15%         6.82x    4/15   n/a

### Conclusions
1. The user's thesis ("fewer parameters => better at restating in fewer words") is
   REFUTED at 0.8B and CONFIRMED at 4B. The 0.8B retains facts perfectly but cannot
   stop; the 4B genuinely compresses and is the only arm that emitted finish=stop.
2. But a 40-line DETERMINISTIC clause extractor matches or beats the 4B at every ratio
   tested (1.97x/14 facts vs the 4B's 2.20x/13), costs no VRAM, no latency, no model,
   and is reproducible. It is in ~/phase5-bench/extractive2.py.
3. At 32k context the 4B costs 3.6 GiB VRAM on card 1 (fits in the 10.2 GiB free);
   the extractor costs zero.
4. CAVEAT on the metric: it rewards verbatim token survival, which a regex-style
   extractor can game by copying tokens out of context. At aggressive budgets the
   extractive output IS fragment soup ("who owns PAY-4471 Priya Raman in payments"),
   usable as a fact log but not as prose. Read the extractive numbers as "a fact index",
   not as "a readable summary".

### The finding that changes the design
Aggressive compression destroys exactly the facts that make a span FINDABLE. We already
built the retrieval half of this loop (Phase 5 §9: lexical-empty fallback, 57.1% -> 85.7%
recall@3). So the store should keep TWO tiers: a compressed in-context representation
(what render() puts in H_t) and the ORIGINAL text (what recall() searches). Compressing
the only copy would make recall unable to recover what it dropped.

---

## 12. SHIPPED: deterministic compaction replaces the 4B compactor (2026-10-12)

### The decision
The Qwen3.5-4B compactor was given up. It held 7 GiB of VRAM (6.4-7.1 GiB across
32k/131072/185363 ctx) to be UNRELIABLE: across ~8 prompt shapes it expanded its input
(0.44-1.07x) or echoed its own system prompt back as an "Output Format / Constraints"
bullet list. The one good run (2.20x, finish=stop, 13/15 facts) did not reproduce on a
longer, differently-formatted block. The deterministic clause extractor matched or beat it
at EVERY ratio measured (§11) and costs nothing. Trading 7 GiB of VRAM on a coin flip is
the wrong deal for a lane that runs on every retirement.

### What is wired in
`neoharness/compress.py` - model-free, deterministic, ~0 ms per span.
  NEOH_COMPACT_RATIO     = 0.40   (compacted span <= 40% of the original)
  NEOH_COMPACT_MIN_CHARS = 240    (below this, do not bother; the span already fits H_t)
Clause-level selection, not sentence-level: one sentence carries several facts, so dropping
a sentence drops all of them (§11: sentence selection lost 8/15 at 3.35x).

`LTMStore` gained a TWO-TIER store:
  entry["text"]   the ORIGINAL - the only copy, and what recall() searches
  entry["index"]  compact(text) - a LOSSY side-channel, what render() puts in H_t
  stats() exposes stored_chars vs compacted_chars.

Compressing the only copy would leave recall unable to find what compaction discarded.

### Verification
  real retired span          1589 -> 647 chars (2.46x), keeps the load-bearing facts
  end-to-end store           stored 1677 / compacted 735 (2.28x)
  recall@3 on the 20-span set: 12/14 = 85.7% - IDENTICAL to pre-compaction
  the Vault query still retrieves the Vault span (recall reads e["text"])
  166 harness tests pass (11 new in tests/test_compress.py)

### Tests are mutation-checked
Six mutants, six caught: recall-queries-the-index (3 fail), render-uses-original (1),
original-overwritten-in-place (3), budget-ignored (3), compaction-no-op (4), and
reading-order-not-restored (1 - this one SURVIVED the first round and needed its own test,
because score order is still deterministic and the determinism law cannot see it).

### Open
The compression RATIO is a dial, not a constant, and it is currently fixed at 0.40. At that
setting the index keeps ~75% of load-bearing facts; the rest are covered by recall. If a
workload ever needs more facts in-context, raise NEOH_COMPACT_RATIO - it costs context, and
recall is what makes the loss safe.

---

## 13. Context window promoted to 196608 (measured, not guessed)

### Why not 131072
131072 was HALF the model's trained window. flash-next's GGUF says
arch=qwen4exp, 48 layers, 2 KV heads, head_dim 256, rope_dim 64, **ctx=262144**.
So 196608 is native range and needs NO --rope-scaling.

### The measurement (needle in a haystack: fact planted at the START of a long
### prompt, asked for at the END)
  max-context  kv-resident   prompt served   needle      RAM avail   VRAM free
  131072       32768 (25%)   -               -           16,131 MiB   9,804/10,739
  196608       98304 (50%)   182,118 tok     RETRIEVED   12,421 MiB   7,413/8,348
  196608       131072 (67%)  boot only       -           ~11 GiB      6,412/7,346
  262144       65536 (25%)   213,318 tok     RETRIEVED   ~7 GiB       7,721/8,624

196608 at 50% residency is the knee: +50% window over 131072 for ~3.5 GiB less
host RAM, VRAM still comfortable. 262144 also retrieves but leaves ~7 GiB of RAM
on a box that also runs the embedder, the reranker and this harness.

### kv-resident IS "put the RAM cache in VRAM"
--kv-resident N keeps N cells on-GPU and streams the rest to pinned RAM. Measured
across the table above: residency 25% -> 16.1 GiB RAM free, 50% -> 12.4, 67% ->
~11 with no further RAM gain. VRAM-resident KV is still accounted against the
host, so past ~50% it buys VRAM rather than RAM. It is NOT the expert cache:
the cache is 6144 slots / 11.07 GiB in every configuration tested.

### PROMOTED, verified on the live service
  --max-context 196608   --kv-resident 98304   --kv int8   --lookup-chain 7
  boot 24 s, short completion OK, needle test on :8080 = 182,118 tokens (93% of
  window) RETRIEVED, prefill 1,998 tok/s, decode 49.0 tok/s, 84.4% of KV reads
  from VRAM. RAM 11,221 MiB after the long prefill; VRAM 7,123/8,027 MiB free.
  Backup: ~/phase1-restore/strata-coder-iq1_m.json.pre-promote196608-*

### Guarded by laws (tests/test_strata_context_config.py, 6 mutants all caught)
window == the measured knee | residency == 50% of window | --kv quantised |
window <= the trained 262144 | no --rope-scaling forced | the arithmetic fits in
host RAM so a doubling fails loudly instead of as swap pressure weeks later.

### Caveats
  - verified with ONE needle shape and a repetitive filler. Retention across many
    adversarial 190k-token shapes is NOT measured.
  - RAM sits at ~11.2 GiB available under an 182k load, down from 16.1. That is
    the first thing to move if anything else grows on the box.
  - 262144 remains validated but unpromoted. Re-measure RAM before going there.
