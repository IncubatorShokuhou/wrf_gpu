"""GPUWRF_NOAH_URBAN_SOIL_PARAMS (V033-W2, default 0): WRF's hard-coded URBAN soil parameters vs the wired pristine WRF.

WRF TRANSFER_MP_PARAMETERS (module_sf_noahmpdrv.F:1694-1700) sets, for URBAN_FLAG columns (:1462-1467), SMCMAX = .45,
SMCREF = .42, SMCWLT = .40, SMCDRY = .40 and CSOIL = 3.E6 before FRZX (:1702-1706). Every consumer sees them: THERMOPROP
HCPCT/TDFCND -> TSNOSOI, PHASECHANGE, RSURF/PSI, WATER/INFIL/SOILWATER. The port used the soil-table values, so its urban
HCPCT was about 30 % low (larger soil diurnal swing, colder urban nights; FINAL33V d03 urban TSLB/T2 signature).
Audit: proofs/noahmp/urban_soil/URBAN_AUDIT.md.

Oracle: the WIRED pristine NOAHMP_SFLX driver (proofs/noahmp/noahmp_offline_driver.F90, NOAHMP_WIRE_SOIL=1
NOAHMP_WIRE_FICEOLD=1, REAL4), real CPU-WRF 0227 d03 columns (36 urban + 24 non-urban low-land, state 2026-02-28_00,
held forcing, dt 6 s); fixture from proofs/noahmp/urban_soil/make_fixture.py. Rule PRE-REGISTERED (MAILBOX b-thompson
2026-10-07T04:3xZ, before any run): key ON -> per-layer TSLB <= 2 REAL ulp of the WRF value at every step for urban AND
non-urban, plus identical L2/L3 changed-step counts per column (WRF's 1/2-ulp stagnation). Key OFF must fail on urban L1-L3,
and non-urban OFF == ON bitwise. Mutants must fail by assertion (E202).
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import jax

FIX = Path(__file__).with_name("fixtures")
NSTEPS = 600
TSLB_ULP = 2.0
SH2O_ABS = 2.0e-5   # companion bound for SH2O and SMC (not in the pre-registered TSLB rule): measured key-on worst 9.9e-6


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    jax.clear_caches()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        jax.clear_caches()


@pytest.fixture(scope="module")
def fx():
    cols = json.loads((FIX / "urban_soil_wired.json").read_text())["columns"]
    with np.load(FIX / "urban_soil_wired_steps.npz") as z:
        wrf = {k: z[k][:NSTEPS] for k in ("tslb", "sh2o", "smc")}
    urb = np.array(["_urban_" in c["name"] for c in cols])
    frozen = np.array(["_frozen_" in c["name"] for c in cols])
    t0 = np.array([c["state_in"]["stc"][3:] for c in cols], np.float32)
    return {"cols": cols, "wrf": wrf, "urban": urb, "frozen": frozen, "t0": t0}


def run(fx, key, patch=None):
    """noah_mp_step (jit, release Noah precision path) NSTEPS steps; returns TSLB/SH2O (nsteps, ncol, 4)."""
    import test_snow_wrf_order_oracle as T
    from gpuwrf.physics.noahmp import noahmp_driver as drv, water_hydro as wh
    with _env(GPUWRF_NOAH_URBAN_SOIL_PARAMS=key, GPUWRF_NOAHMP_NATIVE_REAL="1"):
        saved = (dict(drv.URBAN_SOIL), drv._urban_soil_energy, wh._urban_soil_on, wh.noahmp_water_hydro, drv.noahmp_water_hydro)
        try:
            if patch:
                patch(drv, wh)
            c = T._build(fx["cols"])
            step = jax.jit(lambda ls: drv.noah_mp_step(ls, c["forcing"], c["static"], c["dt"],
                                                        energy_params=c["ep"], rad_params=c["rp"])[0])
            land, tslb, sh2o, smc = c["land"], [], [], []
            for _ in range(NSTEPS):
                land = step(land)
                h = jax.device_get(land)
                tslb.append(np.asarray(h.tslb)[:, 0, :].T); sh2o.append(np.asarray(h.sh2o)[:, 0, :].T)
                smc.append(np.asarray(h.smois)[:, 0, :].T)
        finally:
            drv.URBAN_SOIL.clear(); drv.URBAN_SOIL.update(saved[0])
            drv._urban_soil_energy, wh._urban_soil_on, wh.noahmp_water_hydro, drv.noahmp_water_hydro = saved[1:]
    out = {k: np.stack(v).astype(np.float32) for k, v in (("tslb", tslb), ("sh2o", sh2o), ("smc", smc))}
    assert all(np.isfinite(a).all() for a in out.values())
    return out


def score(fx, res):
    w = fx["wrf"]["tslb"]
    ulp = np.abs(res["tslb"].astype(np.float64) - w.astype(np.float64)) / np.spacing(np.abs(w)).astype(np.float64)
    chg = lambda a: (np.diff(np.concatenate([fx["t0"][None], a]), axis=0) != 0).sum(0)  # noqa: E731
    return {"ulp": ulp.max(0),                                         # (ncol, 4) worst over steps
            "stagnation_mismatch": (chg(res["tslb"])[:, 2:] != chg(w)[:, 2:]).any(1),
            "sh2o": np.maximum(np.abs(res["sh2o"] - fx["wrf"]["sh2o"]), np.abs(res["smc"] - fx["wrf"]["smc"])).max(0)}


@pytest.fixture(scope="module")
def on(fx):
    return run(fx, "1")


@pytest.fixture(scope="module")
def off(fx):
    return run(fx, None)


def test_fixture_is_urban_and_stagnating(fx):
    assert fx["urban"].sum() >= 20 and (~fx["urban"]).sum() >= 20
    w = fx["wrf"]["tslb"]
    # WRF's deep layers sit in 1/2-ulp stagnation on most columns -> the stagnation count is a sharp discriminator
    assert ((np.diff(w[:, :, 3], axis=0) == 0).mean(0) > 0.9).sum() >= 30
    # frozen-wet urban variants: soil ice present and WRF infiltrates (INFIL frozen-soil term with FRZX active)
    c = [x for x in fx["cols"] if "_frozen_" in x["name"]]
    assert len(c) >= 8 and all(x["state_in"]["smc"][0] - x["state_in"]["sh2o"][0] > 0.1 for x in c)
    assert (fx["wrf"]["smc"][-1, fx["frozen"], 0] > fx["wrf"]["smc"][0, fx["frozen"], 0]).all()


def test_key_on_matches_wired_pristine(fx, on):
    s = score(fx, on)
    assert s["ulp"].max() <= TSLB_ULP, (s["ulp"].max(0), s["ulp"].argmax())
    assert not s["stagnation_mismatch"].any(), np.flatnonzero(s["stagnation_mismatch"])
    assert s["sh2o"].max() <= SH2O_ABS, s["sh2o"].max()


def test_key_off_fails_on_urban_only(fx, on, off):
    s = score(fx, off)
    urb = fx["urban"]
    assert (s["ulp"][urb][:, 1:] > TSLB_ULP).any(1).sum() >= 10        # urban L1-L3 out of bound
    np.testing.assert_array_equal(off["tslb"][:, ~urb], on["tslb"][:, ~urb])
    np.testing.assert_array_equal(off["sh2o"][:, ~urb], on["sh2o"][:, ~urb])
    np.testing.assert_array_equal(off["smc"][:, ~urb], on["smc"][:, ~urb])


MUTANTS = {  # each removes ONE piece of WRF's urban block (table-like value or one side); must fail the oracle
    "csoil_table": lambda drv, wh: drv.URBAN_SOIL.update(csoil=2.0e6),
    "smcmax_table": lambda drv, wh: drv.URBAN_SOIL.update(smcmax=0.439),
    "smcwlt_table": lambda drv, wh: drv.URBAN_SOIL.update(smcwlt=0.066),
    "energy_side_dropped": lambda drv, wh: setattr(drv, "_urban_soil_energy", lambda energy, vt, p: energy),
    "water_side_dropped": lambda drv, wh: setattr(wh, "_urban_soil_on", lambda: False),
    "frzx_before_override": lambda drv, wh: _frzx_from_table(drv, wh),
}


def _frzx_from_table(drv, wh):
    """FRZX formed from the soil-TABLE SMCMAX/SMCREF, i.e. BEFORE the urban override (WRF forms it after, drv:1702-1706)."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(wh.noahmp_water_hydro))
    old = "    frzx_default = _FRZK * (layer_row(smcmax,0) / smcref_safe) * (0.412 / 0.468)\n"
    new = ("    frzx_default = _FRZK * (layer_row(_soil_param(parameters, 'smcmax', static, smc, 0.45),0) / jnp.maximum("
           "layer_row(_soil_param(parameters, 'smcref', static, smc, 0.35),0), 1.0e-12)) * (0.412 / 0.468)\n")
    assert src.count(old) == 1
    ns = dict(wh.__dict__)
    exec(compile(src.replace(old, new), wh.__file__, "exec"), ns)  # noqa: S102
    wh.noahmp_water_hydro = drv.noahmp_water_hydro = ns["noahmp_water_hydro"]


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(fx, name):
    s = score(fx, run(fx, "1", MUTANTS[name]))
    killed = (s["ulp"].max() > TSLB_ULP) or s["stagnation_mismatch"].any() or (s["sh2o"].max() > SH2O_ABS)
    assert killed, (name, s["ulp"].max(), s["sh2o"].max())
