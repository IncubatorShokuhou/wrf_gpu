"""SLUCM (sf_urban_physics=1) kernel vs pristine-WRF oracle savepoints.

Oracle: proofs/v040/urban_slucm (unmodified module_sf_urban.F driven by a verbatim
extract of module_sf_noahmpdrv.F:noahmp_urban; 3 URBPARM tables, fp64 + fp32 builds):
* base set: 25 regimes = 18 real WN3 0227 d03 urban columns + 7 stress regimes, 30 steps;
* branch set (rv-urban F2): 12 regimes, 60 steps, executing the admitted branches the base set
  missed (IMP_SCHEME=2 roof/road saturation, canopy-wind ZR+2 >= ZA else-branch).

Gates (pre-registered with the lane, o2-urban notes):
* fp64: per field over every regime/step, max |JAX - WRF| <= 1e-10 * max|WRF|.
* fp32: per field, max |JAX32 - WRF32| <= 3 * max |WRF32 - WRF64| + 4 ulp(max|WRF|)
  (the oracle's own REAL rounding spread).
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "proofs" / "v040" / "urban_slucm"
sys.path.insert(0, str(PROOF))

import slucm_parity as sp  # noqa: E402

from gpuwrf.physics import urban_slucm as us  # noqa: E402


def _check_fp64(errs):
    bad = {k: e for k, e in errs.items()
           if e["nonfinite"] or e["max_abs"] > 1e-10 * max(e["ref_absmax"], 1e-30)}
    return bad


REGIME_SETS = ["", "_branch"]


@pytest.mark.parametrize("regimes", REGIME_SETS)
@pytest.mark.parametrize("tbl", ["default", "alt1", "alt2"])
def test_fp64_step_parity(tbl, regimes):
    errs = sp.run("fp64", tbl, regimes=regimes)
    assert len(errs) >= 55
    bad = _check_fp64(errs)
    assert not bad, {k: (e["max_abs"], e["ref_absmax"]) for k, e in bad.items()}


@pytest.mark.parametrize("regimes", REGIME_SETS)
@pytest.mark.parametrize("tbl", ["default", "alt1", "alt2"])
def test_fp32_step_parity_within_real_rounding(tbl, regimes):
    errs = sp.run("fp32", tbl, regimes=regimes)
    z32 = np.load(sp.SAVE / f"slucm_fp32_{tbl}{regimes}.npz")
    z64 = np.load(sp.SAVE / f"slucm_fp64_{tbl}{regimes}.npz")
    bad = {}
    for k, e in errs.items():
        assert e["nonfinite"] == 0, k
        if k not in z32.files:
            continue
        noise = float(np.nanmax(np.abs(z32[k][:, 1:] - z64[k][:, 1:])))
        ulp = float(np.spacing(np.float32(e["ref_absmax"])))
        if e["max_abs"] > 3.0 * noise + 4.0 * ulp:
            bad[k] = (e["max_abs"], noise, ulp)
    assert not bad, bad


def test_oracle_regimes_cover_admitted_branches():
    """E154 census of the regimes, checked on the oracle OUTPUTS (not only the inputs).

    Base set: day/night, rain > 1 mm/h, calm (UA<1), CHS/CQS2 below the 0.01 limit, the wrfinput
    FRC_URB2D path and (alt1) the IMP_SCHEME=2 reservoirs.  Branch set (rv-urban F2): the canopy-wind
    else-branch (ZR+2 >= ZA -> UC = UA/2) with dz8w1 = 19.1 m on the other side, and IMP_SCHEME=2 roof,
    wall and road storage at saturation (DREL = DENGIMP).  Per rv-urban gcov, the remaining unexecuted
    lines of urban() belong only to refused features (NUDAPT, slucm_distributed_drag, GROPTION=1,
    CH_SCHEME=1, IRI_SCHEME=1, the SHADOW block).
    """
    z = np.load(sp.SAVE / "slucm_fp64_default.npz")
    rain_mm_h = z["in_rainbl"] / z["in_dt"] * 3600.0
    assert (rain_mm_h > 1.0).sum() >= 2
    assert (z["in_swdown"] > 0).sum() >= 6 and (z["in_swdown"] == 0).sum() >= 6
    assert (np.hypot(z["in_u1"], z["in_v1"]) < 1.0).any()
    assert (z["in_chs"] < 0.01).any() and (z["in_cqs2"] < 0.01).any()
    assert np.isclose(z["in_frc_urb"], 0.62).any() and np.isclose(z["in_frc_urb"], 0.9).any()
    assert (z["UC"][:, 1:] > 5.0).any() or (np.hypot(z["in_u1"], z["in_v1"]) > 15).any()
    a1 = np.load(sp.SAVE / "slucm_fp64_alt1.npz")
    assert (a1["DRELR"][:, 1:] > 0).any() and (a1["DRELG"][:, 1:] > 0).any()
    # branch set
    b = np.load(sp.SAVE / "slucm_fp64_default_branch.npz")
    names = list(b["regime_names"])
    ua = np.maximum(np.hypot(b["in_u1"], b["in_v1"]), 1.0)
    low = [i for i, n in enumerate(names) if n.startswith("rv_lowza") and b["in_dz8w1"][i] <= 19.0]
    assert len(low) >= 3
    for i in low:
        np.testing.assert_allclose(b["UC"][i, 1:], ua[i] / 2.0, rtol=1e-12)
    tall = names.index("rv_lowza_19p1")
    assert not np.allclose(b["UC"][tall, 1:], ua[tall] / 2.0)
    table = us.parse_urbparm(sp.SAVE / "URBPARM.TBL.alt1", dtype=np.float64)
    a1b = np.load(sp.SAVE / "slucm_fp64_alt1_branch.npz")
    for field, k in (("DRELR", 0), ("DRELB", 1), ("DRELG", 2)):
        assert np.isclose(a1b[field][:, 1:].max(), table.dengimp[k], rtol=1e-12), field
        assert (np.isclose(a1b[field][:, 1:], table.dengimp[k], rtol=1e-12).sum(axis=1) > 10).any(), field


def test_unsupported_urbparm_options_refused(tmp_path):
    src = (sp.SAVE / "URBPARM.TBL.default").read_text()
    for key, val in (("CH_SCHEME", "1"), ("GROPTION", "1"), ("IRI_SCHEME", "1")):
        txt = "\n".join(f"{key}: {val}" if line.startswith(f"{key}:") else line for line in src.splitlines())
        p = tmp_path / "URBPARM.TBL"
        p.write_text(txt + "\n")
        table = us.parse_urbparm(p)
        with pytest.raises(NotImplementedError, match=key):
            us.check_supported_options(table)
    us.check_supported_options(us.parse_urbparm(sp.SAVE / "URBPARM.TBL.default"))


def test_mutants_are_caught(monkeypatch):
    """E39: the parity gate fails when a kernel piece is removed or perturbed."""
    # M1: layer conduction removed
    monkeypatch.setattr(us, "multi_layer", lambda bound, g0, cap, aks, tsl, dz, delt, tslend: tsl)
    assert _check_fp64(sp.run("fp64", "default"))
    monkeypatch.undo()
    # M2: urban fraction blend weight perturbed (table FRC_URB for UTYPE 2)
    def bump_frc(table):
        frc = table.frc_urb.copy()
        frc[1] = 0.85
        return dataclasses.replace(table, frc_urb=frc)
    bad = _check_fp64(sp.run("fp64", "default", mutate=bump_frc))
    assert "HFX" in bad and "TSK" in bad
    # M3: force-restore damping constant perturbed (TS_SCHEME=2 table)
    orig = us.force_restore

    def fr(cap, aks, delt, s, r, h, le, tslend, tsp):
        return orig(cap, aks, delt * 1.0001, s, r, h, le, tslend, tsp)

    monkeypatch.setattr(us, "force_restore", fr)
    assert _check_fp64(sp.run("fp64", "alt2"))


def test_savepoints_load_from_clean_git_archive(tmp_path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    rel = "proofs/v040/urban_slucm"
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", rel], check=True,
                             capture_output=True, text=True).stdout.split()
    need = [f"{rel}/savepoints/slucm_{p}_{t}{r}.npz" for p in ("fp64", "fp32") for t in ("default", "alt1", "alt2")
            for r in ("", "_branch")]
    need += [f"{rel}/regimes.txt", f"{rel}/regimes_branch.txt", f"{rel}/make_branch_regimes.py"]
    need += [f"{rel}/savepoints/URBPARM.TBL.{t}" for t in ("default", "alt1", "alt2")]
    need += [f"{rel}/slucm_parity.py", f"{rel}/urban_oracle.f90", f"{rel}/build_and_run.sh"]
    missing = sorted(set(need) - set(tracked))
    assert not missing, missing
    arch = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", rel], check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=arch.stdout, check=True)
    for regimes in ("", "_branch"):
        errs = sp.run("fp64", "default", savepoint_dir=tmp_path / rel / "savepoints", regimes=regimes)
        assert not _check_fp64(errs)


def _kernel_mutant(old, new):
    """In-process source mutant of physics/urban_slucm.py (no file on disk is modified)."""
    import importlib.util
    import types

    src = Path(us.__file__).read_text()
    assert src.count(old) == 1, old
    name = "gpuwrf.physics._urban_slucm_mutant"
    mod = types.ModuleType(name)
    mod.__file__ = us.__file__
    sys.modules[name] = mod
    try:
        exec(compile(src.replace(old, new), us.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    del importlib
    return mod


BRANCH_MUTANTS = {
    # rv-urban RVUR06: three mutants inside branches only the branch set executes, one control.
    "MA_else_canopy_wind": ("uc = jnp.where(tall, uc_t, ua / c(2.0))", "uc = jnp.where(tall, uc_t, ua / c(3.0))",
                            "default"),
    "MB_roof_saturation_clip": (
        "drelr = jnp.where(drelr <= c(0.0), c(0.0), jnp.where(drelr <= denr, drelr, denr))",
        "drelr = jnp.where(drelr <= c(0.0), c(0.0), jnp.where(drelr <= denr, drelr, denr * c(0.9)))", "alt1"),
    "MD_road_saturation_beta": (
        "betg = jnp.where(drelg <= c(0.0), c(0.0), jnp.where(drelg <= deng, drelg / deng * porg, porg))",
        "betg = jnp.where(drelg <= c(0.0), c(0.0), jnp.where(drelg <= deng, drelg / deng * porg, porg * c(0.9)))",
        "alt1"),
    "MC_control_imp1_rain_beta": ("betr = jnp.where(rain > c(1.0), c(0.7), betr)",
                                  "betr = jnp.where(rain > c(1.0), c(0.71), betr)", "default"),
}


@pytest.mark.parametrize("name", sorted(BRANCH_MUTANTS))
def test_branch_mutants_are_caught(name):
    """rv-urban F2: each branch mutant fails the fp64 gate on the tracked branch set (E39/E202)."""
    old, new, tbl = BRANCH_MUTANTS[name]
    mod = _kernel_mutant(old, new)
    assert not _check_fp64(sp.run("fp64", tbl, regimes="_branch"))     # unmutated gate is clean
    bad = _check_fp64(sp.run("fp64", tbl, regimes="_branch", kernel=mod))
    assert len(bad) >= 5, (name, sorted(bad))
