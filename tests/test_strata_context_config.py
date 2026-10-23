"""Laws for the live Strata CONTEXT WINDOW (docs/ngram-gate-plan.md §13).

These are not style preferences. The window was chosen by measurement, and the
cost of getting it wrong is silent: too large and the engine eats host RAM until
the box starts swapping, and nothing errors.

Measured on .107 (2x Arc Pro B70, 31 GiB host RAM), needle-in-haystack where a
fact is planted at the START of a long prompt and asked for at the END:

  max-context  kv-resident   prompt served   needle   RAM avail   VRAM free
  131072       32768 (25%)   -               -        16,131 MiB   9,804/10,739
  196608       98304 (50%)   182,118 tok     RETRIEVED 12,421 MiB  7,413/8,348
  196608       131072 (67%)  boot only       -        ~11 GiB      6,412/7,346
  262144       65536 (25%)   213,318 tok     RETRIEVED  ~7 GiB      7,721/8,624

196608 at 50% residency is the measured knee: +50% window over the old 131072
for about 3.5 GiB less host RAM, with VRAM still comfortable. 262144 also
works and also retrieves, but it leaves only ~7 GiB of RAM on a box that also
runs the embedder, the reranker and the harness.

262144 is the model's NATIVE trained window (the GGUF says ctx=262144,
rope_dim=64), so neither setting needs --rope-scaling. Anything beyond 262144
would be real extrapolation and is NOT covered by these tests.
"""
import json
from pathlib import Path

import pytest

CONFIG = Path("/home/bbear/Strata/strata-coder-iq1_m.json")

# flash-next: 48 layers x 2 KV heads x 256 head_dim = 49152 elements/token.
# At fp16 that is 96 KiB/token, so 196608 tokens alone would be 18 GiB of K/V.
ELEMENTS_PER_TOKEN = 2 * 48 * 2 * 256


def _cfg():
    if not CONFIG.exists():
        pytest.skip(f"no live Strata config at {CONFIG}")
    return json.loads(CONFIG.read_text())


def _flag(cfg, name):
    args = cfg.get("args") or []
    return args[args.index(name) + 1] if name in args else None


def _ctx(cfg):
    return int(_flag(cfg, "--max-context"))


def _resident(cfg):
    return int(_flag(cfg, "--kv-resident"))


# ---- the values -----------------------------------------------------------

def test_window_is_the_measured_knee():
    """196608, not 262144.

    262144 also passes its needle test, so this is not a correctness law -- it
    is the point where host RAM runs out. If you raise it, re-measure RAM with
    `free -m` under a full-window prefill first; do not raise it by eye.
    """
    assert _ctx(_cfg()) == 196608, (
        "--max-context is the measured RAM knee; see the table in this module's "
        "docstring. Raising it to 262144 costs ~5 GiB of host RAM."
    )


def test_kv_residency_is_half_the_window():
    """kv-resident IS the "put the RAM cache in VRAM" knob.

    Measured: 25% -> 16.1 GiB RAM free, 50% -> 12.4 GiB, 67% -> ~11 GiB with no
    further RAM gain (VRAM-resident KV is still accounted against the host).
    Past ~50% it buys VRAM, not RAM, and starves the expert cache.
    """
    cfg = _cfg()
    assert _resident(cfg) * 2 == _ctx(cfg), (
        f"kv-resident {_resident(cfg)} is not 50% of max-context {_ctx(cfg)}; "
        "measured knee is 98304 of 196608"
    )


def test_kv_cache_is_quantised():
    """fp16 K/V at this window would be ~18 GiB on its own."""
    assert _flag(_cfg(), "--kv") in ("int8", "q8_0", "q4_0"), (
        "--kv must stay quantised; fp16 at 196608 does not fit in host RAM"
    )


def test_window_does_not_exceed_the_trained_length():
    """Beyond 262144 is extrapolation and needs --rope-scaling.

    These tests were validated at 196608 and at 262144, both inside the trained
    window. A config claiming more has left what was measured.
    """
    assert _ctx(_cfg()) <= 262144, (
        "max-context exceeds the model's trained window (262144); that needs "
        "--rope-scaling linear|yarn and has NOT been validated here"
    )


def test_no_rope_scaling_is_forced():
    """Both windows we measured are native, so a rope correction is a sign the
    window moved and the measurement did not follow."""
    assert _flag(_cfg(), "--rope-scaling") is None, (
        "--rope-scaling is set; re-run the needle test before trusting this window"
    )


def test_the_measured_knee_would_fit_in_ram():
    """The arithmetic, so a config edit that doubles the window fails loudly
    instead of failing as swap pressure weeks later."""
    cfg = _cfg()
    per_token = ELEMENTS_PER_TOKEN * 2.0        # fp16 is the worst case
    if _flag(cfg, "--kv") == "q4_0":
        per_token = ELEMENTS_PER_TOKEN * (18 / 32)
    else:
        per_token = ELEMENTS_PER_TOKEN * (34 / 32)
    total_gib = per_token * _ctx(cfg) / 2 ** 30
    assert total_gib < 16.0, (
        f"full K/V would be {total_gib:.1f} GiB; this box has 31 GiB total and "
        "needs room for the embedder, the reranker and the harness"
    )
