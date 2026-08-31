"""Measure our own matrices with the Weight Atlas's metrics.

The atlas repo ships the data but not weight_atlas.py, and the PLE's own
matrices (edge_proj / value_proj — the 5120<->160 n-gram buffer projections)
aren't in the checkpoint anyway. This reimplements the three quantizers in
numpy so the buffer's matrices get the same measured treatment:

- INT8 per-row-channel symmetric
- INT4 group-128 symmetric (along the last axis)
- FP8 e4m3 (finite max 448; NaN encodings excluded)

plus SQNR in dB (10*log10(||W||^2 / ||W - W_hat||^2)) and row_amax_ratio
(max row abs-max / median row abs-max — our definition; the atlas's exact
normalization is unpublished).
"""

from __future__ import annotations

import numpy as np


def sqnr_db(w: np.ndarray, w_hat: np.ndarray) -> float:
    num = float(np.sum(w.astype(np.float64) ** 2))
    den = float(np.sum((w.astype(np.float64) - w_hat.astype(np.float64)) ** 2))
    return float("inf") if den == 0 else 10.0 * np.log10(num / den)


def quant_int8_per_row(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(w / scale), -127, 127) * scale


def quant_int4_g128(w: np.ndarray, group: int = 128) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    rows, cols = w.shape
    pad = (-cols) % group
    wp = np.pad(w, ((0, 0), (0, pad)))
    g = wp.reshape(rows, -1, group)
    scale = np.abs(g).max(axis=2, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(g / scale), -7, 7) * scale
    return q.reshape(rows, -1)[:, :cols]


def _e4m3_grid() -> np.ndarray:
    vals = [0.0]
    for exp in range(0, 16):
        for mant in range(8):
            if exp == 0:
                v = (mant / 8.0) * 2.0**-6  # subnormal
            elif exp == 15 and mant == 7:
                continue  # NaN encoding — excluded (finite max 448)
            else:
                v = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
            vals.append(v)
    return np.array(sorted(set(vals)))


_E4M3 = _e4m3_grid()


def quant_fp8_e4m3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    sign = np.sign(w)
    a = np.clip(np.abs(w), 0, 448.0)
    idx = np.searchsorted(_E4M3, a)
    idx = np.clip(idx, 1, len(_E4M3) - 1)
    lo, hi = _E4M3[idx - 1], _E4M3[idx]
    nearest = np.where(a - lo <= hi - a, lo, hi)
    return sign * nearest


def row_amax_ratio(w: np.ndarray) -> float:
    amax = np.abs(np.asarray(w, dtype=np.float64)).max(axis=1)
    med = float(np.median(amax))
    return float(amax.max() / med) if med > 0 else 1.0


def measure(w: np.ndarray) -> dict:
    """The atlas metric set for one 2-D matrix."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {w.shape}")
    return {
        "shape": list(w.shape),
        "sqnr_int8_ch": sqnr_db(w, quant_int8_per_row(w)),
        "sqnr_int4_g128": sqnr_db(w, quant_int4_g128(w)),
        "sqnr_fp8_e4m3": sqnr_db(w, quant_fp8_e4m3(w)),
        "row_amax_ratio": row_amax_ratio(w),
        "std": float(w.std()),
        "absmax": float(np.abs(w).max()),
    }
