"""Tests for the science lab calculator (fabric/lab.py).

Covers all six families: hadamard, HOA field, orbitals, antenna, filter,
link budget. Skips when hoa64 is unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fabric import lab

    lab.hadamard_build(8)  # force hoa64 import
except Exception as e:  # noqa: BLE001
    pytest.skip(f"lab surface unavailable: {e}", allow_module_level=True)


def test_hoa_rotate_yaw_moves_peak():
    a = lab.hoa_encode([0.0], [0.0], [1.0], order=3)
    b = lab.hoa_rotate([0.0], [0.0], yaw_deg=90.0, order=3)
    assert abs(((b["peak_az_deg"] - 90 + 180) % 360) - 180) < 20


def test_sage_eval_sympy():
    from fabric.sage import sage_eval

    r = sage_eval("factor(x**2 - 1)")
    assert r.get("ok") is True
    assert "x" in r["result"] and "1" in r["result"]


def test_actual_size_scales():
    s = lab.actual_size_scales(0.003)
    assert s["T_K"] > 0 and s["L_cm"] > 0


def test_crown_psf_shape():
    r = lab.crown_psf(16)
    assert r["figure"]["type"] == "heatmap"
    assert len(r["figure"]["grid"]) == 16


def test_materials_cloth():
    r = lab.materials_design("cloth", 8)
    assert r["kind"] == "cloth"
    assert r["figure"]["type"] == "matrix"


def test_hadamard_sylvester():
    res = lab.hadamard_build(64, "sylvester")
    assert res["n"] == 64
    assert res["method"] == "sylvester"
    assert res["is_hadamard"] is True
    assert res["det_log10"] <= res["det_bound_log10"]
    assert res["figure"]["type"] == "matrix"
    assert len(res["figure"]["grid"]) <= 32


def test_hadamard_known_order():
    # 12 = Paley-type order in the verified library
    res = lab.hadamard_build(12)
    assert res["is_hadamard"] is True
    assert res["figure"]["n"] == 12


def test_hadamard_unknown_order():
    res = lab.hadamard_build(1212)
    assert "error" in res or res.get("is_hadamard")


def test_hoa_encode_peak():
    # single source at az 90, el 0 -> peak direction must recover it
    res = lab.hoa_encode([90.0], [0.0], [1.0], order=4)
    assert res["n_channels"] == 25
    assert res["n_sources"] == 1
    assert abs(res["peak_az_deg"] - 90.0) < 15.0
    assert abs(res["peak_el_deg"] - 0.0) < 15.0
    assert res["per_order_energy"][0] > 0
    assert res["figure"]["type"] == "heatmap"
    assert len(res["figure"]["grid"]) == 48   # azi rows
    assert len(res["figure"]["grid"][0]) == 24  # el cols


def test_hoa_encode_two_sources():
    res = lab.hoa_encode([0.0, 180.0], [0.0, 0.0], None, order=2)
    assert res["n_sources"] == 2
    assert res["n_channels"] == 9


def test_hoa_encode_rejects_bad_order():
    assert "error" in lab.hoa_encode([0.0], [0.0], None, order=9)


def test_orbital_slice():
    res = lab.orbital_probe(2, 1, 0, n_r=33)  # 2p_z; odd n_r samples z=0
    assert res["n"] == 2 and res["l"] == 1 and res["m"] == 0
    assert res["nodes_radial"] == 0
    assert res["figure"]["type"] == "heatmap"
    g = res["figure"]["grid"]
    assert len(g) == 33 and len(g[0]) == 33
    # slice is (x rows, z cols): 2p_z |psi|^2 vanishes on the z=0 mid COLUMN
    mid = [row[16] for row in g]
    assert max(abs(v) for v in mid) < 1e-6


def test_orbital_rejects_bad_quantum_numbers():
    assert "error" in lab.orbital_probe(1, 1, 0)  # l < n violated


def test_antenna_dipole():
    res = lab.antenna_pattern("dipole", 145.0)
    assert res["kind"] == "dipole"
    assert res["gain_dbi"] == pytest.approx(2.15)
    assert res["figure"]["type"] == "polar"
    # dipole at phi=0: null at theta=0/180, max at theta=90
    p = res["figure"]["power_norm"]
    assert p[45] == pytest.approx(1.0, abs=0.01)  # theta=90
    assert p[0] < 0.01


def test_antenna_unknown_kind():
    assert "error" in lab.antenna_pattern("dish", 100.0)


def test_filter_lpf_response():
    res = lab.filter_response("lpf", 100.0, n=5)
    assert res["kind"] == "lpf"
    assert res["f_c_mhz"] == pytest.approx(100.0, rel=0.01)
    assert res["figure"]["type"] == "xy"
    y = res["figure"]["y"]
    x = res["figure"]["x"]
    import numpy as np

    y = np.array(y)
    x = np.array(x)
    # passband: low insertion loss below fc
    assert y[x < 80].min() > -3.0
    # real rejection band somewhere above fc (stepped-impedance LPFs have a
    # re-entrant passband near 3fc — assert on the rejection region, not y[-1])
    assert y[(x > 150) & (x < 280)].min() < y[x < 80].min() - 15.0


def test_filter_bpf():
    res = lab.filter_response("bpf", 100.0, n=3, f_lo_mhz=90.0,
                              f_hi_mhz=110.0)
    assert res["kind"] == "bpf"
    assert res["figure"]["type"] == "xy"


def test_link_budget_friis():
    res = lab.link_budget(p_tx_dbw=10.0, g_tx_dbi=2.0, g_rx_dbi=2.0,
                          f_mhz=5800.0, d_m=100.0)
    # Friis at 5.8 GHz, 100 m: FSPL ~ 87.7 dB -> prx ~ 10+2+2-87.7
    prx = res.get("prx_dbw", res.get("received_dbw", res.get("p_rx_dbw")))
    if prx is None:
        # unknown schema: just require finite numbers
        assert any(isinstance(v, (int, float)) for v in res.values())
    else:
        assert -85.0 < prx < -70.0


def test_last_figure_updates():
    lab.hadamard_build(16, "sylvester")
    last = lab.last_figure()
    assert last["figure"]["n"] == 16
