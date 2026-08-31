"""Long-horizon benchmark v2: state-dependent ledger under forced elision.

v1 taught us the model shortcuts a linear ledger with one grep batch and
mental arithmetic (and gets it wrong). v2 makes batching useless: each
transaction's effect DEPENDS on the running totals —

  txn file: "seed: N, rule: if the running total of <C> is even, add N to
             <C>; otherwise add N to <D>"

so the only way to be right is to carry all three running totals through
every step. Both arms get the same small context budget (fit_chars,
keep_turns=4): early turns are elided mid-run in both. The raw arm's totals
live only in elided chat history; the BPE arm can externalize them via
commit/note and read them back from the surviving H_t render. Verified
against exact ground truth from the generator.

Usage: PYTHONPATH=. python scripts/bench_longhorizon.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

from neoharness import Agent, BPEState, Harness
from neoharness.compactor import ContextCompactor
from neoharness.goals import GoalRunner
from neoharness.llm import DutyLLM
from neoharness.skills import SkillStore
from neoharness.tools import make_core_tools

MODEL = os.environ.get(
    "BENCH_MODEL", str(Path.home() / "models/flashnext/flashnext-q4km-pleq6k.gguf"))
CATEGORIES = ("tools", "parts", "materials")
N_TXNS = 20
FIT_CHARS = 6000
KEEP_TURNS = 4


def make_workspace(ws: Path, seed: int = 7) -> dict:
    rng = random.Random(seed)
    truth = {c: 0 for c in CATEGORIES}
    for i in range(1, N_TXNS + 1):
        cat = CATEGORIES[rng.randrange(3)]
        dest = CATEGORIES[(CATEGORIES.index(cat) + 1) % 3]
        n = rng.randint(1, 9)
        # simulate the rule against the running truth
        target = cat if truth[cat] % 2 == 0 else dest
        truth[target] += n
        (ws / f"txn_{i:02d}.txt").write_text(
            f"seed: {n}, rule: if the running total of {cat} is even, "
            f"add {n} to {cat}; otherwise add {n} to {dest}.\n"
        )
    return truth


GOAL = (
    "Process txn_01.txt through txn_20.txt IN ORDER. Each file says: "
    "'seed: N, rule: if the running total of CATEGORY is even, add N to "
    "CATEGORY; otherwise add N to OTHER'. Start all three totals "
    "(tools, parts, materials) at 0 and apply each file's rule to your "
    "RUNNING totals. Your context window is LIMITED — earlier turns WILL be "
    "elided, so after each file keep your current totals in a place that "
    "survives elision (for example a harness commit or note), because you "
    "cannot recover them from the history once elided. After txn_20, write "
    'totals.json containing exactly {"tools": T, "parts": P, "materials": M} '
    "and emit <done>totals written</done>."
)


def check(ws: Path, truth: dict) -> bool:
    f = ws / "totals.json"
    if not f.exists():
        return False
    try:
        got = json.loads(f.read_text())
    except json.JSONDecodeError:
        return False
    return all(got.get(c) == truth[c] for c in CATEGORIES)


def run_arm(llm, ws: Path, truth: dict, use_bpe: bool, max_steps: int,
            max_tokens: int) -> dict:
    compactor = ContextCompactor(fit_chars=FIT_CHARS, keep_turns=KEEP_TURNS)
    harness = Harness(llm, state=BPEState())
    agent = Agent(llm, harness=harness, tools=make_core_tools(ws),
                  compactor=compactor, max_steps=max_steps, use_bpe=use_bpe)
    tok0 = llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]
    t0 = time.time()
    if use_bpe:
        ep = GoalRunner(agent, SkillStore(harness.state)).run_goal(GOAL).episode
    else:
        ep = agent.run(GOAL)
    tok1 = llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]
    totals_file = ws / "totals.json"
    got = None
    if totals_file.exists():
        try:
            got = json.loads(totals_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "arm": "bpe" if use_bpe else "raw",
        "verified": bool(ep.solved and check(ws, truth)),
        "solved": ep.solved,
        "totals_written": got,
        "steps": len(ep.steps),
        "harness_calls": ep.harness_calls,
        "invalid": ep.invalid_calls,
        "elisions": len(compactor.transfers),
        "tokens": tok1 - tok0,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seeds", default="7,11,23")
    ap.add_argument("--out", default="data/bench_longhorizon.json")
    args = ap.parse_args()

    llm = DutyLLM(model=MODEL, max_tokens=args.max_tokens)
    report = {"seeds": {}, "arms": []}
    for seed in [int(s) for s in args.seeds.split(",")]:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            truth = make_workspace(ws, seed=seed)
            report["seeds"][seed] = {"truth": truth}
            print(f"seed {seed} ground truth: {truth}", flush=True)
            for use_bpe in (False, True):
                (ws / "totals.json").unlink(missing_ok=True)
                r = run_arm(llm, ws, truth, use_bpe, args.max_steps,
                            args.max_tokens)
                r["seed"] = seed
                report["arms"].append(r)
                print(json.dumps(r), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    # verdict table
    for arm in ("raw", "bpe"):
        rows = [r for r in report["arms"] if r["arm"] == arm]
        ok = sum(r["verified"] for r in rows)
        print(f"{arm}: verified {ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
