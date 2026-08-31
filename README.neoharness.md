# NeoHarness

A better harness for the duty LLM (Qwen3.8-27B on vLLM), fusing:

- **EvoHarness-RL** ([arXiv:2608.05446](https://arxiv.org/html/2608.05446v1)) —
  Belief / Progress / Experience (**BPE**) as policy-facing harness state
  `H_t = (B_t, P_t, E_t)`, and the compact harness action protocol
  `{track, commit, recall, note}`.
- **BICAMERAL_FRAMEWORK** (`~/M1Multitronic/docs/BICAMERAL_FRAMEWORK.md`) —
  the two-mind topology: GPU1 ("subconscious") captures, gates, and stores;
  GPU0 ("conscious" duty model) decides. Transport contract: hidden-state
  capture → `dct_topk` compact → **row_cos ≥ 0.95 gate** → n-gram buffer.
- **M1Multitronic n-gram code** (`~/M1Multitronic`, imported, never vendored) —
  `writable_ngram.py` (.mv2 store), the `ple-buffer-worker.py` JSON-line
  protocol (store/activate/stats, merge|replace, persist, freq-pinned
  eviction), and the `afi` spectral package (correctness oracle for the gate).

## The core mapping: BPE = the XYZ vectors of the n-gram buffer

Each harness state component is one coordinate axis of the buffer:

| BPE component | Axis | Buffer session | Harness actions |
|---|---|---|---|
| Belief (environment estimate) | **X** | `bpe-x` | `track[object\|world]` |
| Progress (subgoal record) | **Y** | `bpe-y` | `commit[subgoal \| status]` |
| Experience (skill bank) | **Z** | `bpe-z` | `recall[query]`, `note[kind \| insight]` |

The policy sees BPE as **rendered text** spliced into the prompt; every
harness write is also mirrored into the n-gram buffer as **latent edge
vectors** on that axis's session (render → hidden capture → row_cos gate →
worker `store` → `activate`). The harness state thus lives in both
representations at once — inspectable for the policy, and riding the PLE
refresh path into the duty model's decode for the subconscious lane.

## Layout

- `neoharness/bpe.py` — the three stores + JSON persistence (`H_t` render).
- `neoharness/actions.py` — `<action>verb[arg]</action>` parse + dispatch;
  env-action passthrough; axis routing table.
- `neoharness/vectors.py` — the XYZ lane: `capture_hidden` (vLLM
  `/hidden_states`), `dct_topk` gate (row_cos ≥ 0.95), `BufferWorker`
  (ple-buffer-worker pipe protocol), `XYZBuffer.store_axis`.
- `neoharness/consolidate.py` — epoch-boundary experience consolidation
  (add/update/remove, usage-count preservation, pruning = harness evolution).
- `neoharness/harness.py` — the episode loop, cost ledger (harness vs env
  calls share one budget), JSONL trajectory recorder.
- `neoharness/llm.py` — duty-model chat client, model-card THINKING sampling
  (temp 1.0, top_p 0.95, top_k 20).
- `neoharness/atlas.py` — Weight Atlas analyzer (~/atlas checkout,
  `ATLAS_JSONL` override): per-layer report, PLE/XYZ tap-candidate ranking,
  measured §6.5 INT4 gate table, and `cache_map()` — the buffer residency
  map (`data/buffer_cache_map.json`): per-layer tier (hot/warm/cold),
  recommended dct_topk rate (0.75/0.5/0.25), and prefetch order. A
  weight-side PRIOR: it sets defaults; the live row_cos gate still decides
  what ships.
- `neoharness/measure.py` — the atlas metrics (INT8-ch / INT4-g128 /
  FP8-e4m3 SQNR, row_amax_ratio) in numpy for OUR OWN matrices
  (edge_proj/value_proj — not in the checkpoint, so not in the atlas).

## Agentic layer (v0.2, Hermes-Agent-style)

Autonomous long-context goal pursuit, modeled on
[Hermes Agent](https://github.com/nousresearch/hermes-agent) and fused with
the BPE substrate:

- `neoharness/agent.py` — the tool-calling loop. Per turn the model may emit
  `<tool>name {"arg": ...}</tool>` (prompt-side convention — the duty 27B has
  no native tool_choice), one `<action>` BPE block, and `<done>summary</done>`.
  Memory nudge every `NEOH_NUDGE_EVERY` (8) steps. `delegate` tool spawns a
  nested subagent (depth-capped).
- `neoharness/tools.py` — core sandboxed toolset: `read_file`, `write_file`,
  `list_dir`, `grep` (jailed to `NEOH_WORKSPACE`), `run_shell` (allowlist
  `NEOH_SHELL_ALLOW`, default ls/cat/grep/find/echo/pwd). Output capped.
- `neoharness/compactor.py` — context fit under `NEOH_FIT_CHARS`: elided
  turns are offered to the XYZ latent lane (`store_axis("x")`) before being
  dropped — M1's n-gram compaction-transfer contract, not a text summary.
- `neoharness/skills.py` — Hermes skills as procedural memory: SKILL.md
  files under `NEOH_STATE_DIR/skills/`, mirrored into the Experience store
  (Z axis) so `recall[...]` finds them; `create_from_trajectory` distills a
  skill from every solved episode (the learning loop).
- `neoharness/goals.py` — `GoalRunner.run_goal(goal, subgoals)`: seeds
  Progress, runs the Agent, distills a skill on success (blocked + mistake
  note on failure), consolidates at goal end (the only persist=True write).

```python
from neoharness import Agent, GoalRunner, Harness
from neoharness.llm import DutyLLM

llm = DutyLLM(model="<duty-model-path>")
runner = GoalRunner(Agent(llm, Harness(llm)))
result = runner.run_goal("refactor the parser and run the tests",
                         subgoals=["read parser", "apply fix", "run tests"])
print(result.solved, result.episode.cost, result.skill_path)
```

Offline demo: `python examples/goal_demo.py`.

## CLI

```
python -m neoharness "read facts.txt and write a one-sentence brief to brief.txt" \
    --workspace /path/to/ws --model <duty-model-id> [--latent] [--max-steps 40]
```

Streams each step (tool calls, BPE actions, observations) as it happens,
then prints solved/cost/skill and the final harness state. Exit code 0 on
success, 1 on budget exhaustion. `--subgoal` (repeatable) seeds Progress;
`--quiet` prints only the result; state persists across runs in
`NEOH_STATE_DIR`.

## Server (OpenCode / Hermes integration)

```
python -m neoharness serve [--port 8393] [--workspace DIR] [--latent]
```

OpenAI-compatible endpoint at `http://127.0.0.1:8393/v1`:

- `GET /v1/models` → `neoharness-duty`
- `POST /v1/chat/completions` — the last user message runs as a GoalRunner
  goal; the reply is the goal result. `stream: true` emits step deltas as
  SSE chunks. Both tool grammars accepted:
  `<tool>name {args}</tool>` and the model's trained
  `<tool>{"name": ..., "arguments": {...}}</tool>`.

Clients:
- **OpenCode**: provider baseURL `http://127.0.0.1:8393/v1`, model
  `neoharness-duty`.
- **Hermes Agent**: `hermes model` → custom OpenAI-compatible endpoint,
  same URL; or call `python -m neoharness "goal"` from a Hermes skill.

BPE state, skills, and trajectories persist in `NEOH_STATE_DIR`, so
cross-session recall works across both clients. Goals run serially (the
duty GPU is serial anyway).

**Permanence:** `neoharness.service` (systemd `--user`, enabled) serves
`:8393` on boot (linger is on). Override via
`systemctl --user edit neoharness` — e.g. `Environment=NEOH_LATENT=1` for
the XYZ lane; never edit the unit's ExecStart. Logs:
`journalctl --user -u neoharness`.

**Web UI:** `http://127.0.0.1:8393/ui/` — built-in test page (zero deps):
goal box with live SSE step stream, BPE state panel, skills list. APIs:
`GET /api/state`, `GET /api/skills`. LAN-exposed via drop-in
`10-lan.conf` (`--host 0.0.0.0`): `http://<lan-host>:8393/ui/` — no
auth; only bind to a LAN interface on a trusted network.

## Usage

```python
from neoharness import BPEState, Harness
from neoharness.llm import DutyLLM

h = Harness(DutyLLM(model="<duty-model-path>"))      # live duty model on :8000
ep = h.run_episode("clean the mug and put it away", env)
result = h.consolidate(trajectory_summary="...")
print(ep.cost)   # the cost-aware ledger GRPO would optimize
```

Offline demo (no services): `python examples/alfworld_style_loop.py`.

## Env gates

| Var | Default | Meaning |
|---|---|---|
| `NEOH_LATENT` | `0` | `1` enables the XYZ latent lane (touches live services) |
| `NEOH_LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | duty vLLM |
| `NEOH_LLM_MODEL` | `duty` | model name sent in requests |
| `NEOH_ROW_COS_GATE` | `0.95` | transport gate threshold |
| `NEOH_COMPACT_RATE` | `0.5` | dct_topk keep rate |
| `NEOH_STATE_DIR` | `~/.local/state/neoharness` | BPE JSON + trajectories |
| `NEOH_PLE_BUFFER_PYTHON` / `NEOH_PLE_BUFFER_WORKER` | M1 paths | buffer worker spawn |
| `M1_ROOT` | `~/M1Multitronic` | import root for the n-gram code |
| `NEOH_WORKSPACE` | cwd | tool sandbox root (file tools jailed here) |
| `NEOH_SHELL_ALLOW` | `ls,cat,grep,find,echo,pwd` | run_shell allowlist |
| `NEOH_TOOL_MAX_CHARS` | `4000` | per-tool output cap |
| `NEOH_FIT_CHARS` | `48000` | context fit budget (compactor) |
| `NEOH_KEEP_TURNS` | `6` | tail turns never elided |
| `NEOH_NUDGE_EVERY` | `8` | memory-nudge cadence (agent loop) |

## Landmines carried over from the M1 stack

- **Echo**: only harness/env-derived text is stored into the buffer — never
  the duty model's own replies (self-referential decode echo).
- **Live writes never persist**: per-turn axis stores are `persist=False`;
  only `Harness.consolidate()` writes the shard (`persist=True`).
- **Fail-soft**: latent-lane failures (capture, gate, worker) never block the
  text harness; two consecutive worker failures kill the lane for the process.
- **No production restarts**: the latent lane uses the running vLLM
  `/hidden_states` hook and spawns its own worker; it edits nothing in M1.
- v1 ships the SFT-equivalent scaffold + trajectory recording. Cost-aware
  GRPO is future work consuming the recorded JSONL rollouts.

## Tests

`python -m pytest tests -q` — fully offline (fake buffer worker speaks the
real JSON-line protocol; the row_cos gate is tested on accept and reject
paths).

## Benchmarks (live, duty 27B)

`scripts/bench_harness.py` — micro tasks, raw vs BPE vs BPE+atlas-rate:
all arms 4/4 verified; BPE costs ~9x tokens on short tasks (per-turn H_t
re-prefill, 0% prefix-cache hit); atlas latent lane stores at map rate 0.75
with row_cos 0.9958. `scripts/bench_longhorizon.py` — state-dependent
20-transaction ledger under forced elision (fit 6k chars, keep 4 turns).
Multi-seed (7/11/23): raw verified 0/3 (state lost to elision every time);
BPE verified 2/3 with exact ground truth (survived up to 70 elisions via
commit checkpoints). The one BPE failure (seed 23) was tool-call JSON
fragility on large file bodies — hardened since with the verbatim
`<content>` block form. Long-horizon state survival is where the harness
pays for itself; BPE is therefore the default operating mode, and the atlas
cache map supplies the default latent-lane compact rate for the tap layer
(`NEOH_TAP_LAYER`, default 63 → rate 0.75).
