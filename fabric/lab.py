"""Science lab calculator — the remaining hoa64 function families as typed,
JSON-safe calculator calls with figure payloads for the LCARS lab pane.

Families (docs/hoa64-rtu-backend.md, operator 2026-08-23 "all of the other
hoa64 functions"):
  hadamard_build(n, method)   constructions + verification (hoa64.hadamard)
  hoa_encode(az, el, ...)     sound-field encode + analysis (hoa64.encode,
                              hoa64.analysis)
  orbital_probe(n, l, m)      hydrogenic |psi|^2 slice (hoa64.orbitals)
  antenna_pattern(kind, f)    pattern cut + gain (hoa64.em_physics)
  filter_response(kind, fc)   microstrip filter S21 (hoa64.rf_filter)
  link_budget(...)            Friis in lossy media (hoa64.em_physics)

Every result carries an optional `figure` dict {type: matrix|heatmap|polar|
xy, ...}; the LCARS `lab` pane renders the most recent one (GET
/api/lab/last). Heavy search (micromag SA, gerzon ILS, RNN) stays out of
the calculator per the locked host-search decision.
"""

import math

import numpy as np


def _import_hoa64():
    import os
    import sys

    try:
        import hoa64  # noqa: F401

        return True
    except ModuleNotFoundError:
        root = os.environ.get("HOA64_ROOT", os.path.expanduser("~"))
        if os.path.isdir(os.path.join(root, "hoa64")):
            sys.path.insert(0, root)
            import hoa64  # noqa: F401

            return True
        raise


_import_hoa64()

_last = {"title": None, "figure": None}
import threading  # noqa: E402

_last_lock = threading.Lock()


def _store(title, figure):
    with _last_lock:
        _last["title"] = title
        _last["figure"] = figure


def last_figure() -> dict:
    with _last_lock:
        return dict(_last)


def _round_list(a, nd=4):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


# ---------------------------------------------------------------- hadamard

def hadamard_build(n: int, method: str = "auto") -> dict:
    """Build + verify a Hadamard matrix of order n.

    method: "auto" (known-construction lookup, then Sylvester),
            "sylvester" (Kronecker 2^k), "known" (verified library).
    """
    from hoa64 import hadamard as H

    n = int(n)
    if n < 1 or n > 4096:
        return {"error": "n out of range [1, 4096]"}
    mat, used = None, method
    if method in ("auto", "known"):
        mat = H.hadamard_known(n)
        used = "known"
    if mat is None and method in ("auto", "sylvester"):
        mat = H.sylvester(n)
        used = "sylvester"
    if mat is None:
        return {"error": f"no construction available for n={n} ({method})",
                "hint": "810 orders verified in the library; gaps below 2000: 1212, 1940"}
    res = H.check(mat, det=True)
    # figure: sign pattern, block-summed to <=32x32 for large n
    k = max(1, n // 32)
    small = mat[: n // k * k, : n // k * k].reshape(n // k, k, n // k, k)
    fig = small.mean(axis=(1, 3)).astype(float)
    figure = {
        "type": "matrix",
        "n": n,
        "block": k,
        "grid": [[round(float(v), 3) for v in row] for row in fig],
        "caption": f"H({n}) {used}" + (f", block {k}x{k}" if k > 1 else ""),
    }
    _store(f"hadamard H({n}) {used}", figure)
    return {
        "n": n,
        "method": used,
        "is_hadamard": bool(res.get("is_hadamard", res.get("valid", False))),
        "is_sign": bool(res.get("is_sign", True)),
        "det_log10": round(float(res.get("det_log10", H.det_log10(mat))), 3),
        "det_bound_log10": round(float(H.det_bound_log10(n)), 3),
        "figure": figure,
    }


# ---------------------------------------------------------------- hoa field

def hoa_encode(azimuths, elevations, gains=None, order: int = 4) -> dict:
    """Encode point sources into ACN/SN3D HOA coefficients and analyse the
    field: peak direction, per-order energy, equirectangular power map."""
    from hoa64 import analysis, encode

    order = int(order)
    if not 1 <= order <= 7:
        return {"error": "order in [1, 7]"}
    az = np.asarray(azimuths, dtype=np.float64)
    el = np.asarray(elevations, dtype=np.float64)
    if az.shape != el.shape or az.ndim != 1 or az.size == 0:
        return {"error": "azimuths/elevations must be equal-length 1-D arrays"}
    g = None if gains is None else np.asarray(gains, dtype=np.float64)
    hoa = encode.encode_points(az, el, g, degrees=True, max_order=order)
    n_ch = (order + 1) ** 2

    per_order = []
    for o in range(order + 1):
        lo = o * o
        hi = (o + 1) ** 2
        per_order.append(round(float(np.sum(hoa[lo:hi] ** 2)), 5))

    azi, elv, power = analysis.directional_power(
        hoa, n_azi=48, n_el=24, max_order=order)
    paz, pel, pval = analysis.peak_direction(
        hoa, n_azi=48, n_el=24, max_order=order, degrees=True)
    pmax = float(power.max())
    grid = [[round(float(v) / max(pmax, 1e-30), 3) for v in row] for row in power]

    figure = {
        "type": "heatmap",
        "grid": grid,
        "caption": (f"HOA-{order} field power, peak az {paz:.0f}° el {pel:.0f}°"
                    f" ({len(az)} source{'s' if len(az) > 1 else ''})"),
        "xlabel": "azimuth 0-360°", "ylabel": "elevation +90..-90°",
    }
    _store(f"HOA-{order} field", figure)
    return {
        "order": order,
        "n_channels": int(n_ch),
        "n_sources": int(az.size),
        "per_order_energy": per_order,
        "peak_az_deg": round(float(paz), 2),
        "peak_el_deg": round(float(pel), 2),
        "peak_rel_db": round(10 * math.log10(max(pval, 1e-30)), 2),
        "figure": figure,
    }


# ---------------------------------------------------------------- orbitals

def orbital_probe(n: int, l: int, m: int, n_r: int = 48) -> dict:
    """Hydrogenic |psi|^2 slice through the y=0 plane (hoa64.orbitals)."""
    from hoa64 import orbitals

    n, l, m = int(n), int(l), int(m)
    n_r = max(16, min(int(n_r), 96))
    try:
        grid = orbitals.orbital_grid(n, l, m, n_r=n_r)
    except Exception as e:  # noqa: BLE001
        return {"error": f"orbital_grid failed: {e}"}
    dens = grid["density"]
    extent = float(grid["extent"])
    mid = n_r // 2
    sl = dens[:, mid, :]  # y = 0 plane (x-z)
    figure = {
        "type": "heatmap",
        "grid": [[round(float(v) / float(sl.max() + 1e-300), 4) for v in row]
                 for row in sl],
        "caption": f"|psi|^2  n={n} l={l} m={m}, y=0 slice, ±{extent:.0f} bohr",
        "xlabel": "x (bohr)", "ylabel": "z (bohr)",
    }
    _store(f"orbital {n}{l}{m}", figure)
    return {
        "n": n, "l": l, "m": m,
        "extent_bohr": round(extent, 1),
        "grid_points": int(n_r ** 3),
        "peak_density": float(sl.max()),
        "nodes_radial": int(n - l - 1),
        "figure": figure,
    }


# ---------------------------------------------------------------- antennas

_ANTENNA_KINDS = ("dipole", "monopole", "loop", "patch", "helix", "yagi")


def antenna_pattern(kind: str, f_mhz: float, phi_deg: float = 0.0,
                    n_theta: int = 91) -> dict:
    """Pattern cut of an em_physics antenna model on a theta sweep."""
    from hoa64 import em_physics

    kind = str(kind).lower()
    if kind not in _ANTENNA_KINDS:
        return {"error": f"kind must be one of {_ANTENNA_KINDS}"}
    builder = getattr(em_physics, f"build_{kind}")
    ant = builder(float(f_mhz) * 1e6)
    thetas = np.linspace(0.0, 180.0, int(n_theta))
    p = ant["pattern"](np.deg2rad(thetas), math.radians(float(phi_deg)))
    p = np.asarray(p, dtype=np.float64)
    p = p / max(float(p.max()), 1e-30)
    pdb = 10 * np.log10(np.maximum(p, 1e-6))
    front = float(p[0])
    back = float(p[-1]) if kind != "dipole" else float(
        p[int(len(p) * 0.75)])
    figure = {
        "type": "polar",
        "theta_deg": _round_list(thetas, 2),
        "power_norm": _round_list(p, 5),
        "caption": (f"{kind} @ {f_mhz} MHz, φ={phi_deg}°, "
                    f"{ant.get('gain_dbi', '?')} dBi"),
    }
    _store(f"antenna {kind} {f_mhz} MHz", figure)
    dims = {k: round(float(v), 4) for k, v in ant.get("dimensions_m", {}).items()}
    z_in = ant.get("z_in_ohm")
    return {
        "kind": kind,
        "f_mhz": float(f_mhz),
        "gain_dbi": ant.get("gain_dbi"),
        "dimensions_m": dims,
        "polarization": ant.get("polarization"),
        "z_in_ohm": (None if z_in is None else
                     f"{float(np.real(z_in)):.1f}{float(np.imag(z_in)):+.1f}j"),
        "front_to_back_ratio_db": round(10 * math.log10(
            max(front, 1e-9) / max(back, 1e-9)), 2),
        "bandwidth_frac": ant.get("bandwidth_frac"),
        "notes": ant.get("notes", "")[:220],
        "figure": figure,
    }


# ---------------------------------------------------------------- filters

_FILTER_KINDS = ("lpf", "hpf", "bpf", "bsf")


def filter_response(kind: str, f_c_mhz: float, n: int = 5,
                    proto: str = "butterworth", f_lo_mhz: float = None,
                    f_hi_mhz: float = None, eps_r: float = 4.4,
                    h_mm: float = 1.6) -> dict:
    """Microstrip stepped-impedance filter design + S21 sweep."""
    from hoa64 import rf_filter as F

    kind = str(kind).lower()
    if kind not in _FILTER_KINDS:
        return {"error": f"kind must be one of {_FILTER_KINDS}"}
    fc = float(f_c_mhz) * 1e6
    kw = dict(eps_r=float(eps_r), h_m=float(h_mm) * 1e-3)
    if kind == "lpf":
        d = F.design_lpf(fc, n=int(n), proto=proto, **kw)
    elif kind == "hpf":
        d = F.design_hpf(fc, n=int(n), proto=proto, **kw)
    elif kind == "bpf":
        d = F.design_bpf(float(f_lo_mhz) * 1e6, float(f_hi_mhz) * 1e6,
                         n=int(n), proto=proto, **kw)
    else:
        d = F.design_bsf(float(f_lo_mhz) * 1e6, float(f_hi_mhz) * 1e6,
                         n=int(n), proto=proto, **kw)
    sw = F.sweep(d, n_points=121)
    fmhz = np.asarray(sw["f_hz"]) / 1e6
    s21 = np.asarray(sw["s21_db"], dtype=np.float64)
    figure = {
        "type": "xy",
        "x": _round_list(fmhz, 3),
        "y": _round_list(np.clip(s21, -60, 5), 2),
        "caption": (f"{kind.upper()} n={n} {proto}, fc {f_c_mhz} MHz, "
                    f"εr {eps_r}"),
        "xlabel": "MHz", "ylabel": "S21 dB",
    }
    _store(f"filter {kind} {f_c_mhz} MHz", figure)
    comps = d.get("components", [])
    return {
        "kind": kind,
        "order": int(n),
        "prototype": proto,
        "f_c_mhz": round(float(d["f_c"]) / 1e6, 3),
        "n_sections": len(d.get("sections", [])),
        "n_components": len(comps),
        "figure": figure,
    }


# ---------------------------------------------------------------- link budget

def link_budget(p_tx_dbw: float, g_tx_dbi: float, g_rx_dbi: float,
                f_mhz: float, d_m: float, medium: str = "air") -> dict:
    """Friis link budget in a (possibly lossy) medium (hoa64.em_physics)."""
    from hoa64 import em_physics

    res = em_physics.link_budget(float(p_tx_dbw), float(g_tx_dbi),
                                 float(g_rx_dbi), float(d_m),
                                 float(f_mhz) * 1e6, medium=medium)
    out = {}
    for k, v in res.items():
        out[k] = round(float(v), 3) if isinstance(v, (int, float)) else v
    return out


# ---------------------------------------------------------------- HOA decode / rotate

def hoa_decode(azimuths, elevations, gains=None, order: int = 4) -> dict:
    """Encode sources then sample the field at those directions (round-trip)."""
    from hoa64 import decode, encode

    order = int(order)
    if not 1 <= order <= 7:
        return {"error": "order in [1, 7]"}
    az = np.asarray(azimuths, dtype=np.float64)
    el = np.asarray(elevations, dtype=np.float64)
    if az.shape != el.shape or az.size == 0:
        return {"error": "azimuths/elevations must match"}
    g = None if gains is None else np.asarray(gains, dtype=np.float64)
    hoa = encode.encode_points(az, el, g, degrees=True, max_order=order)
    samp = decode.decode_directions(hoa, az, el, degrees=True, max_order=order)
    return {
        "order": order,
        "n_channels": int((order + 1) ** 2),
        "samples": _round_list(samp, 5),
        "azimuths": _round_list(az, 2),
        "elevations": _round_list(el, 2),
    }


def hoa_rotate(azimuths, elevations, yaw_deg: float, pitch_deg: float = 0.0,
               roll_deg: float = 0.0, gains=None, order: int = 4) -> dict:
    """Encode a scene, Wigner-D rotate, re-analyse peak direction."""
    from hoa64 import analysis, encode, rotate

    order = int(order)
    az = np.asarray(azimuths, dtype=np.float64)
    el = np.asarray(elevations, dtype=np.float64)
    g = None if gains is None else np.asarray(gains, dtype=np.float64)
    hoa = encode.encode_points(az, el, g, degrees=True, max_order=order)
    rot = rotate.rotate_yaw_pitch_roll(
        hoa, yaw=float(yaw_deg), pitch=float(pitch_deg),
        roll=float(roll_deg), degrees=True, max_order=order)
    paz, pel, _ = analysis.peak_direction(rot, n_azi=48, n_el=24, max_order=order)
    azi, elv, power = analysis.directional_power(rot, n_azi=48, n_el=24, max_order=order)
    pmax = float(power.max())
    figure = {
        "type": "heatmap",
        "grid": [[round(float(v) / max(pmax, 1e-30), 3) for v in row] for row in power],
        "caption": f"HOA-{order} after yaw {yaw_deg}° pitch {pitch_deg}°",
    }
    _store("HOA rotate", figure)
    return {
        "order": order,
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "peak_az_deg": round(float(paz), 2),
        "peak_el_deg": round(float(pel), 2),
        "figure": figure,
    }


# ---------------------------------------------------------------- FDTD / materials / scales / crown

def fdtd_probe(f_mhz: float = 150.0, medium: str = "air",
               n: int = 16, max_steps: int = 40) -> dict:
    """Small Yee FDTD dipole probe (hoa64.fdtd). n≤24, steps≤80."""
    from hoa64 import fdtd

    n = int(max(16, min(int(n), 24)))
    max_steps = int(max(8, min(int(max_steps), 80)))
    res = fdtd.fdtd_run(float(f_mhz) * 1e6, medium=medium, n=n,
                        max_steps=max_steps, frame_every=max_steps, pol_viz=False)
    info = res.get("info") or {}
    frame = res.get("final_frame") or {}
    sl = frame.get("e_mid_xy")
    if sl is None:
        sl = frame.get("e_mid_xz")
    if sl is None:
        sl = [[0]]
    sl = np.asarray(sl)
    if sl.ndim != 2:
        sl = np.atleast_2d(sl)
    mx = float(np.max(np.abs(sl))) or 1.0
    step = max(1, sl.shape[0] // 32)
    small = sl[::step, ::step]
    figure = {
        "type": "heatmap",
        "grid": [[round(float(v) / mx, 3) for v in row] for row in small],
        "caption": f"FDTD |E| mid-plane {medium} {f_mhz} MHz n={n}",
    }
    _store("FDTD probe", figure)
    out = {
        "f_mhz": float(f_mhz),
        "medium": medium,
        "n": n,
        "steps": info.get("steps_run"),
        "dt_s": info.get("dt_s"),
        "dx_m": info.get("dx_m"),
        "alpha_theory": info.get("alpha_theory"),
        "decay_measured": info.get("decay_measured"),
        "figure": figure,
    }
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 6)
    return out


def materials_design(kind: str = "cloth", order: int = 8) -> dict:
    """H.8 flux-tile materials layout (cloth/touchpad/metamaterial)."""
    from hoa64 import materials

    try:
        d = materials.design(kind, int(order))
        H = materials.normalize(materials.load_H(int(order), "sylvester"))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    figure = {
        "type": "matrix",
        "grid": [[int(v) for v in row] for row in H.tolist()],
        "caption": f"materials {kind} H({order})",
    }
    _store(f"materials {kind}", figure)
    layout = d.get("layout") or {}
    stats = d.get("stats") or {}
    return {
        "kind": kind,
        "order": int(order),
        "n_rects": len(layout.get("rects") or []),
        "n_pads": len(layout.get("pads") or []),
        "bbox": layout.get("bbox"),
        "fill": stats.get("fill"),
        "figure": figure,
    }


def actual_size_scales(eps: float = 0.003) -> dict:
    """Press chemistry-fraction scales (hoa64.actual_size)."""
    from hoa64 import actual_size as A

    try:
        s = A.scales(float(eps))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    out = {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
           for k, v in s.items()}
    return out


def crown_psf(n: int = 32) -> dict:
    """Crown-aperture PSF magnitude (hoa64.crown.psf). Closed-form, no search."""
    from hoa64 import crown

    n = int(max(8, min(int(n), 64)))
    h = np.abs(crown.psf(n))
    mx = float(h.max()) or 1.0
    figure = {
        "type": "heatmap",
        "grid": [[round(float(v) / mx, 3) for v in row] for row in h],
        "caption": f"crown PSF |h| n={n}",
    }
    _store("crown PSF", figure)
    return {"n": n, "peak": round(mx, 5), "figure": figure}
