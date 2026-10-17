"""GPUWRF_NOAH_QSFC_WRF (V033-W2, default 0): Noah-MP land QSFC as WRF writes grid%QSFC, vs the wired pristine WRF.

WRF NOAHMP_SFLX passes ONE QSFC argument through ENERGY: VEGE_FLUX sets the canopy-air value (lsm:4073), then BARE_FLUX
overwrites it with the bare-ground 0.622*ESTG*RHSUR/(PSFC-0.378*ESTG*RHSUR) (lsm:4435); ENERGY's Q1 FVEG blend (lsm:2300)
is a local (lsm:645) that is never exported. The driver writes QSFC1D back as grid%QSFC (noahmpdrv:1244), and the next
SFCLAY_mynn reads it over land (module_sf_mynn.F:531-559: THVGB = THGB*(1+EP1*QSFC)). The port wrote the Q1 blend
(energy.py) -> vegetated daytime QSFC about 2 g/kg low, surface theta_v about .4 K low in MYNN-SL. Urban QSFC is reset
after ENERGY to QFX/(RHOAIR*CH)+QAIR (lsm:1063) in both. Q1 has no other consumer in Noah (BARE_FLUX's entry value is
overwritten on its first iteration), so the key changes QSFC only.

Oracle: the WIRED pristine NOAHMP_SFLX (REAL4, NOAHMP_WIRE_SOIL=1 NOAHMP_WIRE_FICEOLD=1) on real CPU-WRF R0227 d03 DAY
land columns (2026-02-28_12: 45 vegetated, 36 urban, barren; held forcing, dt 6 s); fixture from
<USER_HOME>/wrf_gpu2_lanes/b-thompson/Q2P/make_fixture.py. Rule PRE-REGISTERED (Q2P/RULE.md addendum A2, MAILBOX
b-thompson 2026-10-07T06:0xZ): key ON -> QSFC <= 1e-6 kg/kg at step 1 and <= 1e-5 at every step on every column, and
|column-mean signed| <= 2e-6; key OFF must fail on vegetated columns only; mutants must fail by assertion (E202).
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import jax

FIX = Path(__file__).with_name("fixtures")
NSTEPS = 120
STEP1, ALL, MEAN = 1.0e-6, 1.0e-5, 2.0e-6
FIELDS = ("qsfc", "q2", "hfx", "lh", "grdflx", "tsk", "t2", "tslb")


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
    cols = json.loads((FIX / "qsfc_wrf_day.json").read_text())["columns"]
    with np.load(FIX / "qsfc_wrf_day_steps.npz") as z:
        wrf = {k: z[k][:NSTEPS] for k in ("qsfc", "fveg")}
    urban = np.array([c["vegtyp"] == 13 for c in cols])
    veg = (wrf["fveg"][0] > 0) & ~urban
    return {"cols": cols, "wrf": wrf, "urban": urban, "veg": veg}


MUTANTS = {  # each replaces ONE piece of WRF's QSFC write-back; source patch of noahmp_energy_canopy (key ON)
    "canopy_qsfc_wins": ('if not _qsfc_wrf() else _qsfc_export(bf["qsfc"])\n',
                         'if not _qsfc_wrf() else _qsfc_export(jnp.where(use_veg, vf["qsfc"], bf["qsfc"]))\n'),
    "urban_override_dropped": ("    qsfc = jnp.where(urban, qfx / (rhoair_top * jnp.maximum(ch, MPE)) + forcing.qair, qsfc)\n",
                               "    qsfc = qsfc\n"),
}


def run(fx, key, mutant=None):
    """noah_mp_step (jit, release Noah precision path) NSTEPS steps; per step QSFC + the other coupler outputs."""
    import inspect
    import textwrap

    import test_snow_wrf_order_oracle as T
    from gpuwrf.physics.noahmp import energy as en, noahmp_driver as drv
    with _env(GPUWRF_NOAH_QSFC_WRF=key, GPUWRF_NOAHMP_NATIVE_REAL="1"):
        saved = (en.noahmp_energy_canopy, drv.noahmp_energy_canopy)
        try:
            if mutant:
                old, new = MUTANTS[mutant]
                src = textwrap.dedent(inspect.getsource(en.noahmp_energy_canopy))
                assert src.count(old) == 1
                ns = dict(en.__dict__)
                exec(compile(src.replace(old, new), en.__file__, "exec"), ns)  # noqa: S102
                en.noahmp_energy_canopy = drv.noahmp_energy_canopy = ns["noahmp_energy_canopy"]
            c = T._build(fx["cols"])

            def one(ls):
                land, fl = drv.noah_mp_step(ls, c["forcing"], c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"])
                return land, {"qsfc": fl.qsfc, "q2": fl.q2, "hfx": fl.hfx, "lh": fl.lh, "grdflx": fl.grdflx,
                              "tsk": fl.tsk, "t2": fl.t2, "tslb": land.tslb}
            step = jax.jit(one)
            land, rec = c["land"], {k: [] for k in FIELDS}
            for _ in range(NSTEPS):
                land, r = step(land)
                for k, v in jax.device_get(r).items():
                    rec[k].append(np.asarray(v).reshape(-1) if k != "tslb" else np.asarray(v)[:, 0, :].T.reshape(-1))
        finally:
            en.noahmp_energy_canopy, drv.noahmp_energy_canopy = saved
    out = {k: np.stack(v) for k, v in rec.items()}
    assert all(np.isfinite(a).all() for a in out.values())  # E200
    return out


def qsfc_err(fx, res):
    d = res["qsfc"].astype(np.float64) - fx["wrf"]["qsfc"].astype(np.float64)
    return {"step1": np.abs(d[0]), "all": np.abs(d).max(0), "mean": np.abs(d.mean(1)).max()}


@pytest.fixture(scope="module")
def on(fx):
    return run(fx, "1")


@pytest.fixture(scope="module")
def off(fx):
    return run(fx, None)


def test_fixture_is_day_vegetated_urban_barren(fx):
    cols = fx["cols"]
    assert fx["veg"].sum() >= 30 and fx["urban"].sum() >= 20 and ((fx["wrf"]["fveg"][0] == 0) & ~fx["urban"]).sum() >= 1
    assert min(c["forcing"]["cosz"] for c in cols) > 0.5 and np.median([c["forcing"]["soldn"] for c in cols]) > 300
    # >= 0, not > 0: two real dry-soil vegetated columns have WRF QSFC == 0 exactly (RHSUR underflow); WRF's MYNN-SL then
    # recomputes a saturated QSFC there (module_sf_mynn.F:532 'QSFC(i).le.0.0'), the legacy Q1 blend hid that branch.
    assert np.isfinite(fx["wrf"]["qsfc"]).all() and (fx["wrf"]["qsfc"] >= 0).all()


def test_key_on_matches_wired_pristine(fx, on):
    e = qsfc_err(fx, on)
    assert e["step1"].max() <= STEP1, (e["step1"].max(), int(e["step1"].argmax()))
    assert e["all"].max() <= ALL, (e["all"].max(), int(e["all"].argmax()))
    assert e["mean"] <= MEAN, e["mean"]


def test_key_off_fails_on_vegetated_only(fx, on, off):
    e = qsfc_err(fx, off)
    assert (e["step1"][fx["veg"]] > 1.0e-4).sum() >= 20          # the Q1 blend: ~1e-3 kg/kg off on vegetated columns
    np.testing.assert_array_equal(off["qsfc"][:, ~fx["veg"]], on["qsfc"][:, ~fx["veg"]])


def test_key_changes_qsfc_only(on, off):
    # QSFC is not consumed inside Noah-MP (BARE_FLUX's entry value is overwritten on its first iteration): every other
    # coupler output and the soil column must be bitwise unchanged.
    for k in FIELDS[1:]:
        np.testing.assert_array_equal(on[k], off[k], err_msg=k)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(fx, name):
    e = qsfc_err(fx, run(fx, "1", name))
    killed = (e["step1"].max() > STEP1) or (e["all"].max() > ALL) or (e["mean"] > MEAN)
    assert killed, (name, e["step1"].max(), e["all"].max(), e["mean"])
