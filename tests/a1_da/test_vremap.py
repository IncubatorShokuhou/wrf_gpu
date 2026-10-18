"""vremap (lane a1-io, task VR): eta refinement, real.exe coordinate/base state, hydrostatic balance, remap gates.

Real-file tests use OUR scratch copy of the 0503 zero-update case (not ALISIOS data; skipped if absent). The
hydrostatic balance is checked against the verbatim real.exe block compiled REAL*8 (pristine oracle, needs gfortran).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import vremap as vr
from gpuwrf.init.real_init.types import RVOVRD
from gpuwrf.io.netcdf_lock import Dataset

CASE = Path("<DATA_ROOT>/wrf_gpu2/scratch/a1-da/cpuwrf/cases_0503/zero_update")
real = pytest.mark.skipif(not (CASE / "wrfinput_d01").is_file(), reason="scratch 0503 zero_update case absent")
FC = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")
ORACLE = Path(__file__).resolve().parents[2] / "proofs/a1_da/io/pristine_hydro/run_oracle.py"


def _read(path, names, index=0):
    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        return {n: np.array(ds[n][index] if ds[n].dimensions[:1] == ("Time",) else ds[n][:]) for n in names}


# --------------------------------------------------------------------------- eta refinement (synthetic)

def test_refine_eta_properties():
    znw = np.array([1.0, 0.99, 0.97, 0.94, 0.90, 0.85, 0.79, 0.72, 0.64, 0.5, 0.3, 0.0])
    z = np.array([0, 60, 180, 380, 650, 1000, 1450, 2000, 2700, 4200, 7000, 15000.0])
    new, info = vr.refine_eta(znw, z, z_lo=400, z_hi=2500, dz_max=100)
    assert all(np.any(new == v) for v in znw), "every original level kept bitwise"
    assert new[0] == 1.0 and new[-1] == 0.0 and np.all(np.diff(new) < 0)
    zn = np.interp(-new, -znw, z)
    band = (zn[1:] > 400) & (zn[:-1] < 2500)
    assert np.diff(zn)[band].max() <= 100 + 1e-9
    assert info["splits_per_layer"][0] == 1 and info["splits_per_layer"][-1] == 1
    assert info["e_vert_new"] == len(new) > len(znw)
    for k, n in enumerate(info["splits_per_layer"]):  # split count = ceil(dz/dz_max) inside the band
        if 0 < k < len(znw) - 2 and z[k + 1] > 400 and z[k] < 2500:
            assert n == math.ceil((z[k + 1] - z[k]) / 100)


# --------------------------------------------------------------------------- real.exe coordinate / base state

@real
def test_vertical_coord_and_base_state_reproduce_real_exe():
    f = _read(CASE / "wrfinput_d01", vr.COORD_1D + vr.COORD_0D + vr.BASE_FIELDS + ("MUB", "HGT"))
    vc = vr.vertical_coord(f["ZNW"], 5000.0, 0.2, 2, reference={n: f[n] for n in vr.COORD_1D})
    for n in vr.COORD_1D + vr.COORD_0D:
        assert np.array_equal(np.asarray(vc[n], np.float32), np.asarray(f[n], np.float32)), n
    vc_raw = vr.vertical_coord(f["ZNW"], 5000.0, 0.2, 2)  # no reference: polynomial C3 within fp32 rounding
    assert np.abs(vc_raw["C3F"].astype(np.float64) - f["C3F"]).max() < 1e-6
    with Dataset(str(CASE / "wrfinput_d01")) as ds:
        consts = vr.base_consts(ds)
    bs = vr.base_state(vc, f["HGT"], consts)
    for n in vr.BASE_FIELDS + ("MUB",):
        assert np.abs(bs[n] - f[n]).max() <= 1e-6 * np.abs(f[n]).max() + 1e-6, n


@real
@pytest.mark.skipif(not FC.is_file(), reason="gfortran absent")
def test_hydrostatic_matches_pristine_real_exe_block():
    spec = importlib.util.spec_from_file_location("a1io_pristine_oracle", ORACLE)
    orc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orc)
    names = vr.COORD_1D + ("P_TOP", "MUB", "MU", "PB", "ALB", "PHB", "T") + vr.MOIST
    f = {k: np.asarray(v, np.float64) for k, v in _read(CASE / "wrfinput_d01", names).items()}
    geo = {n: f[n] for n in vr.COORD_1D} | {"P_TOP": float(f["P_TOP"]), "nz": 44, "MUB": f["MUB"], "PB": f["PB"],
                                             "ALB": f["ALB"], "PHB": f["PHB"]}
    state = {"MU": f["MU"], "T": f["T"], **{m: f[m] for m in vr.MOIST}}
    orc.BOX = (slice(28, 36), slice(48, 60))  # Güímar area, land + ocean
    exe = orc.build(real8=True)
    ref = orc.run_fortran(exe, geo, state, "vremap_test", True)
    box = (slice(None),) + orc.BOX
    thm = (f["T"] + 300.0) * (1.0 + RVOVRD * f["QVAPOR"]) - 300.0
    qtot = sum(f[m] for m in vr.MOIST)
    mine = vr.hydrostatic(thm[box], f["QVAPOR"][box], qtot[box], f["MU"][orc.BOX], f["MUB"][orc.BOX],
                          f["PB"][box], f["ALB"][box], f["PHB"][box], f, float(f["P_TOP"]))
    for n in ("P", "PH", "AL"):
        scale = np.abs(ref[n]).max()
        assert np.abs(mine[n] - ref[n]).max() <= 1e-9 * scale + 1e-9, (n, np.abs(mine[n] - ref[n]).max(), scale)


# --------------------------------------------------------------------------- remap gates

def _compare(a_path, b_path):
    out = {}
    with Dataset(str(a_path)) as a, Dataset(str(b_path)) as b:
        a.set_auto_mask(False)
        b.set_auto_mask(False)
        for n, var in a.variables.items():
            x, y = np.asarray(var[:]), np.asarray(b[n][:])
            vertical = any(d in vr.VERT_DIMS for d in var.dimensions)
            if not vertical:
                assert x.tobytes() == y.tobytes() and var.dtype == b[n].dtype, f"non-vertical {n} not byte-identical"
                continue
            if x.dtype.kind == "f" and x.shape == y.shape:  # refined vertical variables change shape
                out[n] = float(np.abs(x.astype(np.float64) - y).max()) / (float(np.abs(x).max()) or 1.0)
    return out


@real
def test_identity_remap_within_real_rounding(tmp_path):
    znw = _read(CASE / "wrfinput_d01", ("ZNW",))["ZNW"]
    vr.remap_case(CASE, tmp_path / "ident", znw)
    rin = _compare(CASE / "wrfinput_d01", tmp_path / "ident" / "wrfinput_d01")
    for n in ("U", "V", "W", "T", "THM", "QVAPOR", "QCLOUD", "QKE", "CLDFRA") + vr.COORD_1D:
        assert rin[n] == 0.0, n
    assert max(rin.values()) <= 2e-5, sorted(rin.items(), key=lambda kv: -kv[1])[:3]
    rb = _compare(CASE / "wrfbdy_d01", tmp_path / "ident" / "wrfbdy_d01")
    assert all(v == 0.0 for n, v in rb.items() if not n.startswith("PH_")) and max(rb.values()) <= 2e-5


def _refined(tmp_path, monkeypatch=None):
    g = _read(CASE / "wrfinput_d01", ("ZNW", "PH", "PHB", "HGT", "LANDMASK"))
    ocean = g["LANDMASK"] == 0
    z = ((g["PH"].astype(np.float64) + g["PHB"]) / vr.G - g["HGT"][None])[:, ocean].mean(axis=1)
    znw_new, info = vr.refine_eta(g["ZNW"], z, dz_max=75.0)
    vr.remap_case(CASE, tmp_path / "ref", znw_new)
    return znw_new, info, tmp_path / "ref"


def check_bdy_coupled_space(src_case, new_case, side="BXS", var="T"):
    """Independent oracle: decouple old T_B with old (c1h·μ_d+c2h), np.interp per column in ln p_d to the new
    levels, recouple with the new coefficients; must equal the written wrfbdy (record 0, mass points)."""
    oi = _read(src_case / "wrfinput_d01", ("C1H", "C2H", "C3H", "C4H", "MUB", "P_TOP"))
    ni = _read(new_case / "wrfinput_d01", ("C1H", "C2H", "C3H", "C4H"))
    with Dataset(str(src_case / "wrfbdy_d01")) as ob, Dataset(str(new_case / "wrfbdy_d01")) as nb:
        ob.set_auto_mask(False)
        nb.set_auto_mask(False)
        b_old = np.asarray(ob[f"{var}_{side}"][0], np.float64)  # (width, nz, ny)
        b_new = np.asarray(nb[f"{var}_{side}"][0], np.float64)
        mu = np.asarray(ob[f"MU_{side}"][0], np.float64)  # (width, ny)
    mub = np.asarray(oi["MUB"], np.float64)[:, : b_old.shape[0]].T  # west strip
    mut = mub + mu
    pt = float(oi["P_TOP"])
    worst = 0.0
    for w in range(b_old.shape[0]):
        for j in range(0, b_old.shape[2], 7):
            m = mut[w, j]
            co = oi["C1H"].astype(np.float64) * m + oi["C2H"]
            cn = ni["C1H"].astype(np.float64) * m + ni["C2H"]
            lo = np.log(oi["C3H"].astype(np.float64) * m + oi["C4H"] + pt)
            ln = np.log(ni["C3H"].astype(np.float64) * m + ni["C4H"] + pt)
            expect = np.interp(-ln, -lo, b_old[w, :, j] / co) * cn
            worst = max(worst, float(np.abs(expect - b_new[w, :, j]).max() / np.abs(expect).max()))
    assert worst <= 2e-6, worst
    return worst


@real
def test_refined_case_gates(tmp_path):
    znw_new, info, case = _refined(tmp_path)
    assert info["e_vert_new"] > info["e_vert_old"] and info["max_dz_in_band_m"] <= 75.0 + 1e-6
    _compare(CASE / "wrfinput_d01", case / "wrfinput_d01")  # asserts non-vertical byte identity
    _compare(CASE / "wrfbdy_d01", case / "wrfbdy_d01")
    o = _read(CASE / "wrfinput_d01", ("T", "QVAPOR", "W", "PH", "PHB", "ZNW"))
    n = _read(case / "wrfinput_d01", ("T", "QVAPOR", "W", "PH", "PHB", "ZNW", "P", "PB"))
    assert n["T"].shape[0] == info["e_vert_new"] - 1 and np.isfinite(n["P"]).all() and (n["P"] + n["PB"] > 0).all()
    kept = [int(np.flatnonzero(n["ZNW"] == v)[0]) for v in o["ZNW"]]
    assert np.array_equal(n["W"][kept], o["W"]), "W at the kept full levels must be the original W"
    zo = (o["PH"].astype(np.float64) + o["PHB"]) / vr.G
    zn = (n["PH"].astype(np.float64) + n["PHB"]) / vr.G
    j, i = 32, 53  # Güímar column (nearest grid column)
    zom, znm = 0.5 * (zo[1:, j, i] + zo[:-1, j, i]), 0.5 * (zn[1:, j, i] + zn[:-1, j, i])
    assert np.abs(np.interp(zom, znm, n["T"][:, j, i]) - o["T"][:, j, i]).max() < 0.1
    assert np.abs(np.interp(zom, znm, n["QVAPOR"][:, j, i]) - o["QVAPOR"][:, j, i]).max() < 2e-4
    assert np.abs(zn[kept] - zo).max() < 5.0, "kept full levels keep their height (hydrostatic rebalance)"
    from gpuwrf.io.namelist_parser import parse_namelist
    nml = parse_namelist(case / "namelist.input")["domains"]
    e_vert = nml["e_vert"][0] if isinstance(nml["e_vert"], list) else nml["e_vert"]
    assert e_vert == info["e_vert_new"] and len(nml["eta_levels"]) == info["e_vert_new"]
    check_bdy_coupled_space(CASE, case)


@real
def test_coupled_space_mutant_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(vr, "_coupling", lambda vc, kind, mu_d: np.ones((len(vc["C1H"]) + (kind == "f"),) +
                                                                          np.shape(mu_d)))
    _, _, case = _refined(tmp_path)
    with pytest.raises(AssertionError):
        check_bdy_coupled_space(CASE, case)
