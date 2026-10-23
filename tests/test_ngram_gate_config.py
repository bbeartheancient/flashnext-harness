"""Laws for the engine's ngram drafting switches (docs/ngram-gate-plan.md §7).

Measured 2026-10-12 on .107, n=24 prompts/arm, paired per-prompt 95% CI:
  --lookup-chain 7 vs chain OFF : verbatim +9.84 tok/s [+6.78,+12.90], 12/12
  depth 3 vs 6                  : verbatim -7.58 tok/s [-10.66,-4.49]
  depth 3 vs 12                 : verbatim -7.67 tok/s [-10.86,-4.47]
  depth 24                      : the chain never fires (0/3 drafts)
  DraftPolicy ON vs OFF         : +4.14 tok/s verbatim (forcing the chain costs more than it earns)

Printing a number is not a test. These pin the three settings the measurement
depends on, so an edit to the live config or an engine flag change that silently
regresses lookup-chain fails here instead of showing up as a slow afternoon.
"""

import json
from pathlib import Path

import pytest

STRATA = Path("/home/bbear/Strata")
CONFIG = STRATA / "strata-coder-iq1_m.json"


def _cfg():
    if not CONFIG.exists():
        pytest.skip(f"no live Strata config at {CONFIG}")
    return json.loads(CONFIG.read_text())


def _flag(cfg, name):
    """The value after `name` in the engine's argv, or None if the flag is absent."""
    args = cfg.get("args") or []
    return args[args.index(name) + 1] if name in args else None


def test_lookup_chain_is_enabled():
    """The payoff of phases 2-3. Off means the measured win is simply absent."""
    assert _flag(_cfg(), "--lookup-chain") == "7", (
        "--lookup-chain must be 7 on the live config; see ngram-gate-plan.md §7"
    )


def test_lookup_chain_depth_stays_at_the_floor():
    """min 3 is the clamp floor AND the measured optimum.

    Raising it monotonically destroys the effect (depth 6: -7.58 tok/s verbatim,
    depth 12: -7.67, depth 24: never fires). The useful signal is at the
    shallowest allowed match, so 'tuning' this upward is a pure regression.
    """
    assert _flag(_cfg(), "--lookup-chain-min") == "3", (
        "--lookup-chain-min must stay at 3; raising it loses ~7.6 tok/s on verbatim work"
    )


def test_draft_policy_is_not_disabled():
    """STRATA_LOOKUP_CHAIN_FIXED=1 forces every chained proposal past the gate.

    Measured: that drops chain acceptance 63.8% -> 31.3%, MTP acceptance
    77.0% -> 67.7%, and costs ~4 tok/s on both families. The engine's own
    DraftPolicy is the gate; it must stay in the loop.
    """
    env = _cfg().get("env") or {}
    assert "STRATA_LOOKUP_CHAIN_FIXED" not in env, (
        "STRATA_LOOKUP_CHAIN_FIXED disables the measured DraftPolicy gate"
    )


def test_drafting_flags_appear_once():
    """A duplicated drafting flag silently shadows its twin's value."""
    args = _cfg().get("args") or []
    for name in ("--lookup-chain", "--lookup-chain-min"):
        assert args.count(name) <= 1, f"{name} appears {args.count(name)} times in args"


def test_speculative_baseline_is_unchanged():
    """Phase 2 changed the chain only. If --spec moves, the A/B is no longer comparable."""
    cfg = _cfg()
    assert _flag(cfg, "--spec") == "4"
    assert _flag(cfg, "--spec-min-p") == "0.5"
