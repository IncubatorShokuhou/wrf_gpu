"""GPUWRF_NOAH_CANWATER_ELAI (V033-W2, default 0): Noah-MP CANWATER capacities from PHENOLOGY vs pristine WRF.

WRF NOAHMP_SFLX hands CANWATER (module_sf_noahmplsm.F:6116 -> :6265) the snow-burial-adjusted ELAI/ESAI of PHENOLOGY
(:1255-1358) and the caller FVEG (:864-875): MAXLIQ = FVEG*CH2OP*(ELAI+ESAI) (:6323) and MAXSNO =
FVEG*6.6*(0.27+46/BDFALL)*(ELAI+ESAI) (:6351). These set FWET and the canopy melt/freeze TV pin. The legacy
port (key unset) uses the unburied LAI+SAI and a local SHDMAX-only FVEG, which differs only on snow-buried
canopies.

Oracle: proofs/noahmp/canwater_elai/make_oracle.py. It compiles VERBATIM PHENOLOGY + CANWATER slices (SHA256 checked)
against the compiled pristine module with the real MODIS MPTABLE (TRANSFER_MP_PARAMETERS), wires them as NOAHMP_SFLX
does and runs 5,544 columns: 11 MODIS categories x SNOWH 0-12 m x liquid/ice/mixed/over-capacity canopy water x TV
below/at/above TFRZ x dew/zero/evaporation. The port side is the real noahmp_phenology_table -> noahmp_water_hydro.
FVEG note: in CANWATER, FVEG only multiplies ELAI+ESAI, and on MODIS categories the PHENOLOGY FVEG differs from
the local one only where ELAI+ESAI == 0. The FVEG half of the fix is therefore wired for caller fidelity but is
output-equivalent; its mutant is pinned as an EQUIVALENT mutant, not counted as a kill (E39/E202).
"""
import inspect
import os
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

FIX = Path(__file__).with_name("fixtures") / "canwater_elai_oracle.npz"
ROOT = Path(__file__).resolve().parents[3]
HVAP, HSUB = np.float32(2.5104e6), np.float32(2.8440e6)
OUT = ("canliq", "canice", "fwet", "tv")
# Release path (native REAL), key on, measured worst vs pristine REAL4 (CPU XLA): canliq/canice 0 ulp, fwet 1, tv 1
# -> bounds in REAL ulps of the WRF value. Key off fails by up to 9.4e6 ulp (fwet) / 7.2e4 ulp (tv) on 804/303 columns.
# Legacy f64 path (GPUWRF_NOAHMP_NATIVE_REAL unset), key on, measured absolute worst: canliq 9.8e-8 mm, canice 9.6e-8 mm,
# fwet 2.6e-7, tv 2.5e-5 K (f64 vs REAL4 rounding; identical with the key off on unburied columns) -> absolute bounds.
BOUND_ULP = {"canliq": 1.0, "canice": 1.0, "fwet": 2.0, "tv": 2.0}
LEGACY_ABS = {"canliq": 2.0e-7, "canice": 2.0e-7, "fwet": 1.0e-6, "tv": 1.0e-4}
PHEN_ULP = 2.0   # PHENOLOGY LAI/SAI/ELAI/ESAI/FVEG: measured <= 1 (REAL) / 1.35 (f64) ulp of the WRF REAL value


def load_fixture():
    with np.load(FIX) as f:
        return {k: f[k] for k in f.files}


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: v for k, v in kv.items() if v is not None})
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
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


def _mutant_canwater(wh, old, new):
    """_canwater with ONE source substitution (exec in the module namespace; caller looks it up by global name)."""
    src = textwrap.dedent(inspect.getsource(wh._canwater))
    assert src.count(old) == 1, (old, src.count(old))
    ns = dict(wh.__dict__)
    exec(compile(src.replace(old, new), wh.__file__, "exec"), ns)  # noqa: S102
    return ns["_canwater"]


def run_port(z, *, key, native, canwater=None, area=None, phen_kw=True):
    """noahmp_phenology_table -> noahmp_water_hydro (jit) on the oracle columns; returns phen + canopy outputs."""
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.physics.noahmp import water_hydro as wh
    from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing

    with _env(GPUWRF_NOAH_CANWATER_ELAI=key, GPUWRF_NOAHMP_NATIVE_REAL=native):
        saved = (wh._canwater, wh._cw_area)
        try:
            if canwater is not None:
                wh._canwater = canwater
            if area is not None:
                wh._cw_area = area
            n = z["in_vegtyp"].size
            fdt = jnp.float32 if native == "1" else jnp.float64
            s2 = lambda a: jnp.asarray(np.asarray(a).reshape(1, n), fdt)  # noqa: E731
            surf = lambda v: jnp.full((1, n), v, fdt)                       # noqa: E731
            soil = lambda v: jnp.full((NSOIL, 1, n), v, fdt)                # noqa: E731
            dzs = jnp.asarray([0.10, 0.30, 0.60, 1.00], fdt)
            static = NoahMPStatic(
                ivgtyp=jnp.asarray(z["in_vegtyp"].reshape(1, n), jnp.int32), isltyp=jnp.full((1, n), 3, jnp.int32),
                xland=surf(1.0), landmask=surf(1.0), lakemask=surf(0.0),
                lu_index=jnp.asarray(z["in_vegtyp"].reshape(1, n), jnp.int32), tbot=surf(285.0),
                dzs=dzs, zsoil=-jnp.cumsum(dzs), lat=surf(28.0), dx_m=1000.0,
                parameters=load_noahmp_parameters(wrf_run_dir()), shdmax=s2(z["in_shdmax"]), shdfac=s2(z["in_shdmax"]))
            zsnso = jnp.concatenate([jnp.zeros((NSNOW, 1, n), fdt),
                                     jnp.broadcast_to(-jnp.cumsum(dzs)[:, None, None], (NSOIL, 1, n))], axis=0)
            land = NoahMPLandState(
                tslb=soil(280.0), smois=soil(0.25), sh2o=soil(0.25), smcwtd=surf(0.25),
                isnow=jnp.zeros((1, n), jnp.int32), tsno=jnp.zeros((NSNOW, 1, n), fdt),
                snice=jnp.zeros((NSNOW, 1, n), fdt), snliq=jnp.zeros((NSNOW, 1, n), fdt), zsnso=zsnso,
                snowh=s2(z["in_snowh"]), sneqv=s2(z["in_snowh"] * 100.0), sneqvo=surf(0.0), tauss=surf(0.0),
                albold=surf(0.65), tv=s2(z["in_tv0"]), tg=s2(z["in_tg"]), tah=s2(z["in_tv"]), eah=surf(800.0),
                canliq=s2(z["in_canliq"]), canice=s2(z["in_canice"]), fwet=surf(0.0), lai=surf(0.0), sai=surf(0.0),
                cm=surf(1e-3), ch=surf(1e-3), t_skin=s2(z["in_tv"]), qsfc=surf(0.004), znt=surf(0.1), emiss=surf(0.97),
                albedo=surf(0.3), sfcrunoff=surf(0.0), udrunoff=surf(0.0))
            zero = surf(0.0)
            forcing = NoahMPForcing(
                sfctmp=s2(z["in_sfctmp"]), sfcprs=surf(9.0e4), psfc=surf(9.0e4), uu=surf(3.0), vv=surf(1.0),
                qair=surf(0.004), qc=zero, soldn=zero, lwdn=surf(300.0), prcpconv=zero, prcpnonc=zero, prcpsnow=zero,
                prcpgrpl=zero, prcphail=zero, cosz=zero, zlvl=surf(28.0),
                julian=jnp.asarray(float(z["in_julian"][0]), fdt), yearlen=jnp.asarray(float(z["yearlen"]), fdt))
            frz = z["in_frz"] == 1
            lath = np.where(frz, HSUB, HVAP).astype(np.float32)          # ENERGY LATHEAV (:2210-2216)
            ecan = (z["in_fcev"] / lath).astype(np.float32)              # CANWATER FCEV/HVAP|HSUB in REAL
            etran = np.maximum(z["in_fctr"] / lath, np.float32(0.0)).astype(np.float32)
            et = NoahMPEtFluxes(ecan=s2(ecan), etran=s2(etran), edir=zero, qseva=zero,
                                btrani=jnp.zeros((NSOIL, 1, n), fdt).at[0].set(1.0), qsnow=zero, qmelt=zero,
                                imelt=jnp.zeros((NSNOW + NSOIL, 1, n), jnp.int32))
            fz = jnp.asarray(frz.reshape(1, n))

            def step(land, et):
                phen = noahmp_phenology_table(land, forcing, static)
                lw = land.replace(lai=phen.lai, sai=phen.sai, tv=s2(z["in_tv"]))  # TV entering CANWATER
                kw = {"phen": phen} if phen_kw else {}
                return phen, wh.noahmp_water_hydro(lw, forcing, static, et, float(z["dt"]), frozen_canopy=fz, **kw)

            phen, out = jax.jit(lambda a, b: step(a, b))(land, et)
            res = {k: np.asarray(getattr(phen, k)).reshape(-1) for k in ("lai", "sai", "elai", "esai", "fveg")}
            res.update({k: np.asarray(getattr(out, k)).reshape(-1) for k in OUT})
            for k, a in res.items():
                assert np.isfinite(a).all(), (k, key, native)
            return res
        finally:
            wh._canwater, wh._cw_area = saved


def excess(z, res, native):
    """Per column and field: |port - WRF| in REAL ulps of the WRF value (release) or absolute (legacy f64)."""
    out = {}
    for v in OUT:
        w = z[f"wrf_{v}"].astype(np.float64)
        d = np.abs(np.asarray(res[v], np.float64) - w)
        if native == "1":
            out[v] = d / np.spacing(np.abs(z[f"wrf_{v}"]).astype(np.float32)).astype(np.float64)
        else:
            out[v] = d
    return out


def fails(z, res, native):
    e = excess(z, res, native)
    bound = BOUND_ULP if native == "1" else LEGACY_ABS
    return {v: int((e[v] > bound[v]).sum()) for v in OUT}


@pytest.fixture(scope="module")
def z():
    return load_fixture()


@pytest.fixture(scope="module")
def buried(z):
    return (z["wrf_elai"] + z["wrf_esai"]) != (z["wrf_lai"] + z["wrf_sai"])


def test_fixture_exercises_snow_burial(z, buried):
    wet = (z["wrf_canliq"] + z["wrf_canice"]) > 0
    full = (z["wrf_elai"] + z["wrf_esai"] == 0) & (z["wrf_lai"] + z["wrf_sai"] > 0)
    assert buried.sum() >= 2000 and (buried & wet).sum() >= 1500 and (full & wet).sum() >= 800
    # the discriminating outputs actually differ between ELAI and LAI capacities on these columns
    assert (buried & wet & (z["wrf_fwet"] > 0)).sum() >= 1000


@pytest.mark.parametrize("native", ["1", "0"])
def test_phenology_inputs_match_pristine(z, native):
    """The producer of CANWATER's ELAI/ESAI/FVEG (port PHENOLOGY + caller FVEG) equals pristine on every column."""
    res = run_port(z, key="1", native=native)
    for v in ("lai", "sai", "elai", "esai", "fveg"):
        d = np.abs(np.asarray(res[v], np.float64) - z[f"wrf_{v}"].astype(np.float64))
        ulp = d / np.spacing(np.abs(z[f"wrf_{v}"]).astype(np.float32)).astype(np.float64)
        assert ulp.max() <= PHEN_ULP, (v, float(ulp.max()))


@pytest.mark.parametrize("native", ["1", "0"])
def test_key_on_matches_pristine_canwater(z, native):
    res = run_port(z, key="1", native=native)
    assert fails(z, res, native) == {v: 0 for v in OUT}, excess_summary(z, res, native)


@pytest.mark.parametrize("native", ["1", "0"])
def test_key_off_is_legacy_and_deviates_only_on_buried_canopies(z, buried, native):
    """Deletion sensitivity of the key itself: OFF fails the oracle on snow-buried canopies and nowhere else."""
    on, off = run_port(z, key="1", native=native), run_port(z, key=None, native=native)
    # SNOWH == 0 -> FB == 0 -> ELAI+ESAI == LAI+SAI: same math; the two jitted programs fuse differently, so XLA:CPU
    # may differ by 1 REAL ulp (E186; measured fwet on 31/693 columns) -> <= 1 ulp (release) / LEGACY_ABS (f64)
    snowfree = z["in_snowh"] == 0.0
    assert snowfree.sum() >= 600
    for v in OUT:
        d = np.abs(np.asarray(off[v], np.float64) - np.asarray(on[v], np.float64))[snowfree]
        if native == "1":
            d = d / np.spacing(np.abs(np.asarray(on[v], np.float32)[snowfree])).astype(np.float64)
        assert d.max() <= (1.0 if native == "1" else LEGACY_ABS[v]), (v, float(d.max()))
    f = fails(z, off, native)
    assert f["fwet"] >= 500, f
    eo = excess(z, off, native)
    bound = BOUND_ULP if native == "1" else LEGACY_ABS
    assert not any(((eo[v] > bound[v]) & ~buried).any() for v in OUT)


MUTANTS = {  # (old, new) in _canwater's key-on branch; each must FAIL the oracle with an assertion (E202)
    "maxliq_lai": ("ch2op * _cw_area(land_state, phen)", "ch2op * (land_state.lai + land_state.sai)"),
    "maxsno_lai": ("1.0e-12)) * _cw_area(land_state, phen)", "1.0e-12)) * (land_state.lai + land_state.sai)"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(z, name):
    from gpuwrf.physics.noahmp import water_hydro as wh
    res = run_port(z, key="1", native="1", canwater=_mutant_canwater(wh, *MUTANTS[name]))
    f = fails(z, res, "1")
    assert f["fwet"] > 0, (name, f)


def test_mutant_elai_only_is_killed(z):
    def elai_only(land_state, phen):
        return jnp.asarray(phen.elai, dtype=jnp.asarray(land_state.lai).dtype)
    res = run_port(z, key="1", native="1", area=elai_only)
    assert fails(z, res, "1")["fwet"] > 0


def test_fveg_half_is_an_equivalent_mutant(z):
    """Local FVEG in place of PHENOLOGY's FVEG changes NO output on MODIS (FVEG only multiplies ELAI+ESAI) -> disclosed,
    not counted as a kill; the wiring stays because it is WRF's caller variable."""
    from gpuwrf.physics.noahmp import water_hydro as wh
    on = run_port(z, key="1", native="1")
    src = textwrap.dedent(inspect.getsource(wh._canwater))
    assert src.count("_cw_fveg(fveg, phen)") == 2
    ns = dict(wh.__dict__)
    exec(compile(src.replace("_cw_fveg(fveg, phen)", "fveg"), wh.__file__, "exec"), ns)  # noqa: S102
    mut = run_port(z, key="1", native="1", canwater=ns["_canwater"])
    for v in OUT:
        np.testing.assert_array_equal(mut[v], on[v], err_msg=v)


def test_key_on_without_phenology_fails_loudly(z):
    with pytest.raises(ValueError, match="CANWATER_ELAI"):
        run_port(z, key="1", native="1", phen_kw=False)


def test_driver_passes_the_steps_phenology_to_water(monkeypatch):
    """Real noah_mp_step: the phen object handed to WATER is the one PHENOLOGY produced in the same step (E85)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_noahmp_coupler import _build
    import gpuwrf.physics.noahmp.noahmp_driver as driver
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
    state, land, static, rad, clock = _build()
    forcing = assemble_noahmp_forcing(state, static, rad, clock, 18)
    seen = {}
    real_phen = driver.noahmp_phenology_table

    def spy_phen(*a, **k):
        seen["phen"] = real_phen(*a, **k)
        return seen["phen"]

    def spy_water(*a, **k):
        seen["water_phen"] = k.get("phen")
        raise RuntimeError("water reached")
    monkeypatch.setattr(driver, "noahmp_phenology_table", spy_phen)
    monkeypatch.setattr(driver, "noahmp_water_hydro", spy_water)
    with pytest.raises(RuntimeError, match="water reached"):
        driver.noah_mp_step(land, forcing, static, 18)
    assert seen["water_phen"] is seen["phen"]


def excess_summary(z, res, native):
    e = excess(z, res, native)
    return {v: float(e[v].max()) for v in OUT}
