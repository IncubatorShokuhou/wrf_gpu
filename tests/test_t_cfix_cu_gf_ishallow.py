"""t-cfix-cu F10: Grell-Freitas (cu=3) runs WRF's Registry default ishallow=0.

WRF passes the namelist ``ishallow`` (Registry default 0; the honour check refuses
any other value) to GFDRV (module_first_rk_step_part1.F:1410 ->
module_cumulus_driver.F:1305); with 0 GF never calls CUP_gf_sh and sets
``cutens=0`` (module_cu_gf_wrfdrv.F:331, :504).  The port's adapter defaulted to 1.

Oracle: the v060 GF column driver against UNMODIFIED pristine WRF 4.7.1 sources,
with ISHALLOW_G3 taken from argv (proofs/t_cfix_cu/gf_ishallow/oracle); its
ishallow=1 output reproduces the committed v060 savepoints byte-for-byte.
Tolerances are the predeclared v060 GPU-batch gate.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "t_cfix_cu" / "gf_ishallow" / "savepoints"
V060 = ROOT / "proofs" / "v060" / "savepoints"
CASES = (1, 2, 3, 4, 5)
FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
REL, FLOOR, RAIN_REL, RAIN_ABS = 2.0e-2, 1.0e-8, 5.0e-2, 1.0e-4  # v060 predeclared


def _load(case, ish, mode=""):
    return json.loads((SAVE / f"gf_case_{case}_ish{ish}{mode}.json").read_text())


def _batched(ishallow, mode=""):
    from gpuwrf.physics import _gf_jax as J

    cases = [_load(c, 0, mode) for c in CASES]  # inputs are identical in both ishallow runs
    kx = int(cases[0]["scalars"]["KX"])
    names = ("T", "QV", "P", "PI", "DZ", "RHO", "U", "V", "W", "RTHBLTEN", "RQVBLTEN")

    def col1(a):
        x = np.zeros(kx + 1)
        x[1:] = np.asarray(a, np.float64)
        return x

    arrs = [jnp.stack([jnp.asarray(col1(d["columns"][n])) for d in cases]) for n in names]
    sc = [d["scalars"] for d in cases]
    vec = lambda k, t=float: jnp.asarray([t(s[k]) for s in sc])  # noqa: E731
    extra = {}
    if mode:  # F18: the WRF caller's held/radiative forcing and PSUR = p8w(1)
        extra = dict(rthften1=jnp.stack([jnp.asarray(col1(d["columns"]["RTHFTEN_IN"])) for d in cases]),
                     rqvften1=jnp.stack([jnp.asarray(col1(d["columns"]["RQVFTEN_IN"])) for d in cases]),
                     rthraten1=jnp.stack([jnp.asarray(col1(d["columns"]["RTHRATEN"])) for d in cases]),
                     psur_pa=vec("PSUR_PA"))
    return J.gfdrv_batched(*arrs, kx, vec("DT"), vec("DX"), vec("HFX"), vec("QFX"),
                           jnp.asarray([int(s["KPBL"]) for s in sc], jnp.int32), vec("XLAND"),
                           jnp.zeros(len(sc)), ishallow_g3=int(ishallow), **extra)


def _failures(out, ish, mode=""):
    bad = []
    for i, c in enumerate(CASES):
        d = _load(c, ish, mode)
        for f in FIELDS:
            got = np.asarray(out[f])[i, 1:]
            ref = np.asarray(d["columns"][f], np.float64)
            err = float(np.max(np.abs(got - ref)))
            if err > FLOOR and err / max(float(np.max(np.abs(ref))), FLOOR) > REL:
                bad.append((c, f, err))
        r_got, r_ref = float(np.asarray(out["RAINCV"])[i]), float(d["scalars"]["RAINCV"])
        if abs(r_got - r_ref) > max(RAIN_ABS, RAIN_REL * abs(r_ref)):
            bad.append((c, "RAINCV", r_got, r_ref))
    return bad


def test_oracle_driver_reproduces_committed_v060_savepoints():
    """E83: the argv-ishallow driver at ishallow=1 equals the old hard-coded-1 driver output."""
    for c in CASES:
        new, old = _load(c, 1), json.loads((V060 / f"gf_case_{c}.json").read_text())
        assert {k: new["columns"][k] for k in old["columns"]} == old["columns"]
        assert {k: new["scalars"][k] for k in old["scalars"]} == old["scalars"]
        assert new["scalars"]["ISHALLOW_G3"] == 1 and _load(c, 0)["scalars"]["ISHALLOW_G3"] == 0
        assert new["scalars"]["FORCING_MODE"] == 0 and _load(c, 0, "_f1")["scalars"]["FORCING_MODE"] == 1


def test_oracle_cases_exercise_the_shallow_switch():
    """E154 census: ishallow=0 must change WRF's own output on shallow-active cases."""
    changed = [c for c in CASES if _load(c, 0)["columns"]["RTHCUTEN"] != _load(c, 1)["columns"]["RTHCUTEN"]]
    assert len(changed) >= 3
    assert all(_load(c, 0)["scalars"]["XMB_SHALLOW"] == 0.0 for c in CASES)
    assert any(_load(c, 1)["scalars"]["XMB_SHALLOW"] > 0.0 for c in CASES)


def test_port_ishallow0_matches_pristine_gfdrv():
    assert _failures(_batched(0), 0) == []


def test_port_wrf_caller_forcing_matches_pristine_gfdrv():
    """F18: held RTHFTEN/RQVFTEN + RTHRATEN in the forced sounding and PSUR = p8w(1)*.01."""
    moved = [c for c in CASES if _load(c, 0, "_f1")["columns"]["RTHCUTEN"] != _load(c, 0)["columns"]["RTHCUTEN"]]
    assert len(moved) >= 3  # E154: the forcing changes WRF's own output
    assert _failures(_batched(0, "_f1"), 0, "_f1") == []
    assert _failures(_batched(0), 0, "_f1")  # deletion: dropping the WRF forcing is caught


def test_ishallow1_kernel_fails_the_ishallow0_oracle():
    """Deletion sensitivity: the old adapter default (shallow on) is caught by the oracle."""
    assert _failures(_batched(1), 0)


def test_operational_gf_slot_runs_ishallow0(monkeypatch):
    """Caller path: the operational stateless cu=3 slot hands GFDRV ishallow_g3=0."""
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    import test_v034_grell_cumulus_wiring as W
    from gpuwrf.coupling import scan_adapters

    seen = []
    real = scan_adapters.gfdrv_batched

    def spy(*args, **kwargs):
        seen.append(kwargs.get("ishallow_g3"))
        return real(*args, **kwargs)

    monkeypatch.setattr(scan_adapters, "gfdrv_batched", spy)
    grid, state, _nml = W._setup(5)
    scan_adapters.CU_STATELESS_SCAN_ADAPTERS[3](state, 60.0, grid)
    assert seen == [0]


def test_ishallow0_port_has_no_shallow_terms():
    """WRF ishallow=0 never calls CUP_gf_sh (module_cu_gf_wrfdrv.F:504): OUTQCs stays 0, so the
    unconditional ``outqcs`` term of RQCCUTEN (:813-832) vanishes, and the shallow diagnostics stay
    0 (:284, :714) -- exact, not merely inside the v060 tolerance."""
    out = _batched(0)
    for f in ("XMB_SHALLOW", "K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW"):
        assert np.all(np.asarray(out[f]) == 0), f
    quiet = 0
    for i, c in enumerate(CASES):
        ref = np.asarray(_load(c, 0)["columns"]["RQCCUTEN"], np.float64)
        if np.all(ref == 0.0):  # WRF: no deep/mid condensate either -> exactly zero
            quiet += 1
            assert np.all(np.asarray(out["RQCCUTEN"])[i, 1:] == 0.0), c
    assert quiet + sum(1 for c in CASES if np.any(np.asarray(_load(c, 0)["columns"]["RQCCUTEN"]))) == len(CASES)
