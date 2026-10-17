"""F20 (caller audit, class B, default path): MYNN gets WRF's bottom-face w and the driver's flux limits.

WRF: first_rk_step_part1.F:1142 ``W=grid%w_2`` -> module_pbl_driver.F:1719 ``w=w`` ->
module_bl_mynnedmf_driver.F:348 ``w1(k) = w(i,k,j)`` (k=kts..kte: bottom face of every layer, surface face
included) -> DMP_mf F90:5855-5879 resolved-w taper ``Psig_w``. The driver also limits its own copies
``hfx1``/``qfx1`` to [-600,1200] W m-2 / [-3e-4,9e-4] kg m-2 s-1 (driver :381-397) before MYNN forms
flt/flqv/fltv (F90:859-876); grid%hfx/qfx stay unlimited.

Gates (pre-registered in .agent/kernel/lanes/t-cfix-mynn.notes.md): pristine PUBLIC ``mynnedmf_driver``
(unmodified objects) on real CPU-WRF PROD d01/d02 + WN3 0227 d03 day/night columns; the logging-only copy
must be BYTE-identical in every public output; taper Psig_w and the limited driver fluxes BYTE; the
release-default kernel MAXMF within 2e-4 m s-1 (+1e-3 rel) of the public MAXMF; caller-path tests on the
product State seam and a 2-step operational physics block. Mutants: tests/v025/t_cfix_mynn/mutants.sh.
"""
from __future__ import annotations

import ast
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import f20_oracle as O  # noqa: E402
from f20_public_kernel import load_sets  # noqa: E402
from gpuwrf.physics import mynn_edmf as E  # noqa: E402
from gpuwrf.physics.surface_layer import mynn_driver_surface_fluxes  # noqa: E402

SETS = ("prod_d01_day", "prod_d01_night", "prod_d02_day", "prod_d02_night", "wn3_d03_day", "wn3_d03_night")
TAPER_SETS = ("prod_d01_night", "prod_d02_day", "prod_d02_night", "wn3_d03_day", "wn3_d03_night")
MAXMF_ABS, MAXMF_REL = 2.0e-4, 1.0e-3
HFX_INJECT = (1500.0, -800.0, 1200.0, -600.0, float(np.nextafter(np.float32(1200), np.float32(2e3))),
              float(np.nextafter(np.float32(-600), np.float32(-2e3))))
QFX_INJECT = (1.2e-3, -5e-4, float(np.float32(9e-4)), float(np.float32(-3e-4)),
              float(np.nextafter(np.float32(9e-4), np.float32(1))), float(np.nextafter(np.float32(-3e-4), np.float32(-1))))


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)


@pytest.fixture(scope="module")
def sets():
    return load_sets()


@pytest.fixture(scope="module")
def oracles(tmp_path_factory):
    root = tmp_path_factory.mktemp("f20_oracle")
    return {"plain": O.build(root / "plain"), "instr": O.build(root / "instr", instrument=True), "root": root}


@pytest.fixture(scope="module")
def runs(sets, oracles):
    cache = {}

    def get(name, kind="plain", inject=False):
        key = (name, kind, inject)
        if key not in cache:
            cols, meta = sets[name]
            if inject:
                cols = {k: v.copy() for k, v in cols.items()}
                cols["hfx"][:6] = np.float32(HFX_INJECT)
                cols["qfx"][6:12] = np.float32(QFX_INJECT)
            work = oracles["root"] / f"run_{name}_{kind}_{int(inject)}"
            cache[key] = (cols, meta, *O.run(oracles[kind], work, cols, meta["dt"], meta["dx"]))
        return cache[key]
    return get


@pytest.mark.parametrize("name", SETS)
def test_logging_copy_is_byte_identical_in_every_public_output(name, runs):
    _, _, plain, _ = runs(name, "plain")
    cols, _, instr, log = runs(name, "instr")
    n = cols["tsk"].shape[0]
    assert sorted(log["F20PSIG"]) == list(range(1, n + 1)) and sorted(log["F20FLUX"]) == list(range(1, n + 1))
    for key in O.OUT2 + O.OUT3:
        assert np.isfinite(plain[key]).all(), key
        assert np.array_equal(_bits(plain[key]), _bits(instr[key])), key


def _port_psig(cols, log, dx, w):
    """Product DMP_mf taper on the caller-supplied w, given WRF's own pblh / Psig_shcu (log)."""
    n, nz = cols["th"].shape
    g = np.array([log["F20PSIG"][i + 1] for i in range(n)])
    dz = jnp.asarray(cols["dz"])
    zw = jnp.concatenate((jnp.zeros_like(dz[:, :1]), jnp.cumsum(dz, axis=-1)), axis=-1)  # phys_mynn_columns.py
    z = jnp.zeros((nz,), jnp.float32)
    s = jnp.zeros((), jnp.float32)

    def one(w1, dz1, zw1, pblh, shcu):
        return E._dmp_setup(z, z, z, z, w1, z + 300.0, z + 300.0, z + 9.0e4, dz1, zw1, z + 1.0, pblh,
                            s + 290.0, s + 1.0, s, s, s, shcu, dx=dx)["psig_w"]

    psig = jax.vmap(one)(jnp.asarray(w, jnp.float32), dz, zw, jnp.asarray(g[:, 3]), jnp.asarray(g[:, 2]))
    return np.asarray(psig), g[:, 1]


@pytest.mark.parametrize("name", SETS)
def test_taper_psig_w_matches_pristine_public_call_bytewise(name, runs):
    cols, meta, _, log = runs(name, "instr")
    nz = cols["th"].shape[1]
    port, wrf = _port_psig(cols, log, meta["dx"], cols["w"][:, :nz])  # what the fixed caller hands MYNN
    assert np.isfinite(port).all() and np.isfinite(wrf).all()
    assert np.array_equal(_bits(port), _bits(wrf)), np.nonzero(_bits(port) != _bits(wrf))[0]


def test_mass_averaged_w_misses_the_pristine_taper(runs):
    """The pre-fix operand (0.5*(w_k+w_k+1)) fails on real terrain columns of every taper set."""
    misses = {}
    for name in TAPER_SETS:
        cols, meta, _, log = runs(name, "instr")
        nz = cols["th"].shape[1]
        port, wrf = _port_psig(cols, log, meta["dx"], 0.5 * (cols["w"][:, :nz] + cols["w"][:, 1:]))
        misses[name] = int(np.count_nonzero(_bits(port) != _bits(wrf)))
    assert all(v >= 1 for v in misses.values()), misses


@pytest.mark.parametrize("name", ("prod_d02_night", "wn3_d03_day"))
def test_driver_flux_limits_match_pristine_public_call_bytewise(name, runs, monkeypatch):
    cols, _, _, log = runs(name, "instr", inject=True)
    n = cols["tsk"].shape[0]
    g = np.array([log["F20FLUX"][i + 1] for i in range(n)])
    assert np.array_equal(g[:6, 0], np.float32([1200, -600, 1200, -600, 1200, -600]))
    assert np.array_equal(g[6:12, 1], np.float32([9e-4, -3e-4, 9e-4, -3e-4, 9e-4, -3e-4]))
    f = lambda a: jnp.asarray(np.asarray(a, np.float32))  # noqa: E731

    def port(limits):
        r = mynn_driver_surface_fluxes(f(cols["hfx"]), f(cols["qfx"]), f(cols["rho"][:, 0]), f(cols["qv"][:, 0]),
                                       f(cols["tsk"]), f(cols["p"][:, 0]), exner=f(cols["exner"][:, 0]),
                                       pbl_limits=limits)
        return np.stack([np.asarray(r.flt), np.asarray(r.flqv), np.asarray(r.fltv)], 1)

    monkeypatch.setenv("GPUWRF_MYNN_PHY_EXNER", "1")  # WRF ex1(kts) = pi_phy (release default BP96)
    limited = port(True)
    assert np.array_equal(_bits(limited), _bits(g[:, 2:]))
    unlimited = port(False)
    wrong = np.nonzero((_bits(unlimited) != _bits(g[:, 2:])).any(1))[0].tolist()
    assert wrong == [0, 1, 4, 5, 6, 7, 10, 11]  # strictly beyond the limits; the limit values themselves pass


def test_release_kernel_maxmf_matches_pristine_public_maxmf(oracles, tmp_path):
    """Whole release-default MYNN column kernel (fresh process, flags resolved like the CLI)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPUWRF_", "XLA_", "JAX_"))}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_FAST_DEFAULTS="1", GPUWRF_JAX_CACHE="0",
               PYTHONPATH=str(ROOT / "src"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out = tmp_path / "kernel.json"
    subprocess.run([sys.executable, str(HERE / "f20_public_kernel.py"), str(oracles["plain"]),
                    str(tmp_path / "work"), str(out)], cwd=ROOT, env=env, check=True, timeout=1500)
    res = json.loads(out.read_text())
    assert res["flags"]["GPUWRF_MYNN_FP32_COLUMNS"] == "1"
    worst, mutant_miss = {}, {}
    for name in SETS:
        r = {k: np.asarray(v, np.float64) for k, v in res[name].items()}
        for key in r:
            assert np.isfinite(r[key]).all(), (name, key)
        bound = MAXMF_ABS + MAXMF_REL * np.abs(r["wrf_maxmf"])
        excess = np.abs(r["face_maxmf"] - r["wrf_maxmf"]) - bound
        worst[name] = float(np.max(np.abs(r["face_maxmf"] - r["wrf_maxmf"])))
        assert (excess <= 0).all(), (name, worst[name])
        mutant_miss[name] = int(np.count_nonzero(np.abs(r["mass_maxmf"] - r["wrf_maxmf"]) > bound))
    assert mutant_miss["wn3_d03_day"] >= 1 and mutant_miss["prod_d02_day"] >= 1, mutant_miss
    (tmp_path / "summary.json").write_text(json.dumps({"worst_face": worst, "mutant_miss": mutant_miss}))


# --- literal DMP_mf taper loop, for the kte face that real columns never reach ------------------------
TAPER_BEGIN = " maxw        = zero\n"
TAPER_END = " Psig_w = min(Psig_w, Psig_shcu)\n"


@pytest.fixture(scope="module")
def literal_taper(tmp_path_factory):
    text = (O.WRF / "phys/module_bl_mynnedmf.F").read_text()
    start = text.index(TAPER_BEGIN)
    stop = text.index(TAPER_END, start) + len(TAPER_END)
    body = text[start:stop]
    assert "do k=1,kte-1" in body and "wpbl = w1(k)" in body and body.count("exit") == 1
    src = """subroutine f20_taper(nz,w1,zw1,dz1,qc1,qc_bl1,cldfra_bl1,pblh,Psig_shcu,psig_out) bind(C)
use iso_c_binding
implicit none
integer(c_int),value :: nz
real(c_float),intent(in) :: w1(nz),zw1(nz+1),dz1(nz),qc1(nz),qc_bl1(nz),cldfra_bl1(nz)
real(c_float),value :: pblh,Psig_shcu
real(c_float),intent(out) :: psig_out
real(c_float),parameter :: zero=0.0,half=0.5,one=1.0
real(c_float) :: maxw,cloud_base,zagl,wpbl,qc_sgs,Psig_w
integer :: k,k50,kte
kte=nz; k50=1
""" + body + """psig_out=Psig_w
end subroutine
"""
    out = tmp_path_factory.mktemp("f20_taper")
    (out / "taper.f90").write_text(src)
    subprocess.run(["timeout", "60", O.FC, "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-ffree-line-length-none",
                    "taper.f90", "-o", "taper.so"], cwd=out, check=True, capture_output=True)
    lib = ctypes.CDLL(str(out / "taper.so"))
    ptr = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    lib.f20_taper.argtypes = [ctypes.c_int, ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_float, ctypes.c_float,
                              ctypes.POINTER(ctypes.c_float)]

    def taper(w1, dz1, pblh, shcu):
        nz = len(dz1)
        dz1 = np.ascontiguousarray(dz1, np.float32)
        zw1 = np.zeros(nz + 1, np.float32)
        for k in range(nz):
            zw1[k + 1] = zw1[k] + dz1[k]
        z = np.zeros(nz, np.float32)
        result = ctypes.c_float()
        lib.f20_taper(nz, np.ascontiguousarray(w1, np.float32), zw1, dz1, z, z, z, pblh, shcu, ctypes.byref(result))
        return np.float32(result.value)
    return taper


def test_taper_reads_faces_kts_to_kte_minus_1_only(sets, literal_taper):
    cols, meta = sets["wn3_d03_day"]
    nz = cols["th"].shape[1]
    w = cols["w"][0, :nz].copy()
    # Real-structured profile rescaled to max |wpbl| = 1.3 (Psig_w = 0.7): every other face leaves the taper
    # partly open, so including face kte (+6 / -4 m/s) would shut it (E29: the branch must matter).
    w = (w * np.float32(1.3 / np.max(np.abs(np.where(w[:nz - 1] < 0, 2 * w[:nz - 1], w[:nz - 1]))))).astype(np.float32)
    cases = []
    for top in (0.0, 6.0, -4.0):
        w[nz - 1] = top  # face kte: inside the zagl cut only for an unreachable pblh
        for pblh in (np.float32(1.0e6), np.float32(800.0)):
            cases.append((w.copy(), pblh))
    dz = cols["dz"][0]
    zw = jnp.concatenate((jnp.zeros((1,), jnp.float32), jnp.cumsum(jnp.asarray(dz))))
    z = jnp.zeros((nz,), jnp.float32)
    s = jnp.zeros((), jnp.float32)
    open_taper = [literal_taper(w1, dz, np.float32(1.0e6), np.float32(1.0)) for w1, pblh in cases if pblh > 1.0e5]
    assert len(set(open_taper)) == 1 and 0.6 < open_taper[0] < 0.8, open_taper  # WRF ignores face kte
    for w1, pblh in cases:
        wrf = literal_taper(w1, dz, pblh, np.float32(1.0))
        port = E._dmp_setup(z, z, z, z, jnp.asarray(w1), z + 300.0, z + 300.0, z + 9.0e4, jnp.asarray(dz), zw,
                            z + 1.0, jnp.float32(pblh), s + 290.0, s + 1.0, s, s, s, jnp.float32(1.0),
                            dx=meta["dx"])["psig_w"]
        assert _bits(port) == _bits(wrf), (float(port), float(wrf), float(w1[nz - 1]), float(pblh))


# --- caller seam: State -> MYNN column view ---------------------------------------------------------
_SMOKE = ROOT / "tests/test_v013_operational_smoke.py"
_spec = importlib.util.spec_from_file_location("f20_smoke_builders", _SMOKE)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _terrain_w(grid, seed=7):
    """Distinct per-face w with a strong (downslope) surface face, like real Tenerife d03 columns."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.6, (grid.nz + 1, grid.ny, grid.nx))
    w[0] = rng.uniform(-3.5, 1.2, (grid.ny, grid.nx))
    return jnp.asarray(w, jnp.float64)


@pytest.mark.parametrize("fp32_columns", ("0", "1"))
def test_caller_hands_mynn_the_bottom_w_faces(fp32_columns, monkeypatch):
    from gpuwrf.coupling import physics_couplers as C

    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", fp32_columns)
    grid = smoke._grid()
    state = smoke._base_state(grid)
    state = state.replace(w=_terrain_w(grid))
    column = C._mynn_column_from_state(state, grid)
    dtype = jnp.float32 if fp32_columns == "1" else state.w.dtype
    w = jnp.asarray(state.w, dtype)
    expect = np.moveaxis(np.asarray(w[:-1]), 0, -1)  # (ny, nx, nz): faces kts..kte
    assert np.asarray(column.w).dtype == expect.dtype
    assert np.array_equal(np.asarray(column.w), expect)
    assert not np.allclose(np.asarray(column.w), np.moveaxis(np.asarray(0.5 * (w[:-1] + w[1:])), 0, -1))


def _noahmp_step(*, grid_backed=False):
    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.coupling.noahmp_surface_hook import noahmp_surface_step
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters

    if not smoke._HAVE_MPTABLE:
        pytest.fail("pristine Noah-MP tables are required for the default-producer caller test")
    grid = smoke._land_grid()
    state = smoke._land_state(grid)
    if grid_backed:  # operational call: grid-backed phy_prep rho/p_hyd/pi_phy need the column dry mass
        state = state.replace(mu_total=jnp.full_like(state.xland, 95000.0), mu=jnp.full_like(state.xland, 95000.0))
    ny, nx = grid.ny, grid.nx
    shape = (ny, nx)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    b = lambda v, s=shape: jnp.broadcast_to(jnp.asarray(v), s)  # noqa: E731
    land = NoahMPLandState(
        tslb=b(295.0, (NSOIL, ny, nx)), smois=b(0.20, (NSOIL, ny, nx)), sh2o=b(0.20, (NSOIL, ny, nx)),
        smcwtd=b(0.20), isnow=jnp.zeros(shape, dtype=jnp.int32), tsno=b(273.0, (NSNOW, ny, nx)),
        snice=b(zero, (NSNOW, ny, nx)), snliq=b(zero, (NSNOW, ny, nx)),
        zsnso=b(jnp.asarray([0.0, 0.0, 0.0, -0.05, -0.25, -0.7, -1.5]).reshape(NSNOW + NSOIL, 1, 1),
                (NSNOW + NSOIL, ny, nx)),
        snowh=zero, sneqv=zero, sneqvo=zero, tauss=zero, albold=b(0.2), tv=b(296.0), tg=b(295.0), tah=b(295.0),
        eah=b(2000.0), canliq=zero, canice=zero, fwet=zero, lai=b(2.0), sai=b(0.5), cm=b(0.01), ch=b(0.01),
        t_skin=b(295.0), qsfc=b(0.01), znt=b(0.05), emiss=b(0.97), albedo=b(0.2), sfcrunoff=zero, udrunoff=zero)
    static = NoahMPStatic(
        ivgtyp=jnp.full(shape, 5, dtype=jnp.int32), isltyp=jnp.full(shape, 6, dtype=jnp.int32),
        xland=jnp.ones(shape), landmask=jnp.ones(shape), lakemask=zero, lu_index=jnp.full(shape, 5, dtype=jnp.int32),
        tbot=b(290.0), dzs=jnp.asarray([0.05, 0.20, 0.45, 0.80]), zsoil=jnp.asarray([-0.05, -0.25, -0.7, -1.5]),
        lat=b(28.0), dx_m=3000.0, parameters=load_noahmp_parameters(smoke._NOAHMP_TABLE_DIR),
        shdmax=b(0.3), shdfac=b(0.3))

    class _Rad:
        soldn = jnp.full(shape, 600.0)
        lwdn = jnp.full(shape, 350.0)
        cosz = jnp.full(shape, 0.6)

    class _Clock:
        julian = 142.0
        yearlen = 365.0

    return noahmp_surface_step(state, land, static, 90.0, radiation=_Rad(), clock=_Clock(),
                               **({"grid": grid} if grid_backed else {}))


# --- driver limits: MYNN-only view; shared handles stay physical for cumulus (critic FIX 23:04Z) -------
# 2x3 land grid: two columns beyond the WRF limits (both signs), the rest inside or exactly at them.
HFX_COLS = ((1500.0, -800.0, 300.0), (1200.0, -600.0, 50.0))
QFX_COLS = ((1.2e-3, -5.0e-4, 1.0e-4), (9.0e-4, -3.0e-4, 2.0e-5))
BEYOND = np.array([[True, True, False], [False, False, False]])


def _extreme_noahmp_step(monkeypatch, *, force_limits=None):
    """Actual default Noah-MP producer with injected physical land HFX/QFX (= grid%hfx/qfx).

    ``force_limits`` overrides the producer's ``pbl_limits`` (None = as shipped): False is the pre-F20
    producer, True the rejected b8fab854ff producer-side limits (leak control)."""
    import gpuwrf.physics.noahmp_coupler as NC
    real_noah, real_flux = NC.noah_mp_step, NC.mynn_driver_surface_fluxes
    calls = []

    def extreme(*args, **kwargs):
        land, flux = real_noah(*args, **kwargs)
        hfx = jnp.asarray(HFX_COLS, flux.hfx.dtype).reshape(flux.hfx.shape)
        qfx = jnp.asarray(QFX_COLS, flux.qfx.dtype).reshape(flux.qfx.shape)
        return land, flux._replace(hfx=hfx, qfx=qfx, lh=jnp.full_like(flux.lh, 3000.0))

    def driver(*args, **kwargs):
        calls.append(dict(kwargs))
        if force_limits is not None:
            kwargs = dict(kwargs, pbl_limits=force_limits)
        return real_flux(*args, **kwargs)

    monkeypatch.setattr(NC, "noah_mp_step", extreme)
    monkeypatch.setattr(NC, "mynn_driver_surface_fluxes", driver)
    monkeypatch.setenv("GPUWRF_MYNN_FLTV_WRF", "1")
    state, _ = _noahmp_step(grid_backed=True)
    assert len(calls) == 1  # the producer really took the WRF MYNN-driver handle branch (E114)
    np.testing.assert_array_equal(np.asarray(state.hfx, np.float32), np.float32(HFX_COLS))
    np.testing.assert_array_equal(np.asarray(state.qfx, np.float32), np.float32(QFX_COLS))
    return state, calls


_CU_KEYS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN", "RUCUTEN", "RVCUTEN")


def _cumulus_flux_operands(cu, state, monkeypatch):
    """HFX/QFX operands the actual cu3/cu6/cu16 adapters hand their kernel (sentinel returns it as RAINCV)."""
    from gpuwrf.coupling import scan_adapters as A

    grid = smoke._land_grid()
    out = {}
    for which in ("hfx", "qfx"):
        if cu == 6 and which == "hfx":
            continue  # CU_TIEDTKE takes QFX only (cumulus_driver.F:1396)
        if cu == 3:  # gfdrv_batched(..., dt_b, dx_b, hfx_b, qfx_b, ...)
            index = 14 if which == "hfx" else 15
            sentinel = lambda *a, i=index, **k: dict({n: jnp.zeros_like(a[0]) for n in _CU_KEYS}, RAINCV=a[i])
            monkeypatch.setattr(A, "gfdrv_batched", sentinel)
            # periodic: WRF's GF lateral band (gf_wrfdrv:264-277, t-cfix-cu F18) would mask every column
            # of this small grid; the operands are per column, so read all of them.
            after = A.gf_adapter(state, 18.0, grid=grid, periodic_x=True, periodic_y=True)
        else:  # (..., QVF0, QVB0|THF0, QFX0, HFX0|XL0, ...)
            index = 14 if which == "qfx" else 15
            # ICUM1: t-cfix-cu c06 ROOT2 ABI (tiedtke_tile_rows reads the row decision); unused by cu=16.
            sentinel = lambda *a, i=index, **k: dict({n: jnp.zeros_like(a[0]) for n in _CU_KEYS}, RAINCV=a[i],
                                                     ICUM1=jnp.bool_(True))
            name, adapter = (("tiedtke_column_jax", A.tiedtke_adapter) if cu == 6
                             else ("ntiedtke_column_jax", A.ntiedtke_adapter))
            monkeypatch.setattr(A, name, sentinel)
            after = adapter(state, 18.0, grid=grid, stepcu=1)
        out[which] = np.asarray(after.rainc_acc, np.float64) - np.asarray(state.rainc_acc, np.float64)
        assert np.isfinite(out[which]).all()
    return out


@pytest.mark.parametrize("cu", (3, 6, 16))
def test_extreme_fluxes_reach_cumulus_unlimited(cu, monkeypatch):
    """WRF hands cumulus the physical grid HFX/QFX (first_rk_step_part1.F:1416); only MYNN's copy is limited."""
    shipped, calls = _extreme_noahmp_step(monkeypatch)
    assert "pbl_limits" not in calls[0]
    got = _cumulus_flux_operands(cu, shipped, monkeypatch)
    with monkeypatch.context() as m:
        reference = _cumulus_flux_operands(cu, _extreme_noahmp_step(m, force_limits=False)[0], m)
    with monkeypatch.context() as m:
        leak = _cumulus_flux_operands(cu, _extreme_noahmp_step(m, force_limits=True)[0], m)
    for which in got:
        assert np.array_equal(got[which], reference[which]), which  # every column, beyond-limit ones included
        assert np.array_equal(leak[which][~BEYOND], reference[which][~BEYOND]), which  # healthy controls
    assert any(not np.array_equal(leak[w][BEYOND], reference[w][BEYOND]) for w in got)  # harness sees a leak
    # The adapters rebuild QFX as State.rhosfc*qv_flux (pbl vs flux density: a pre-existing, column-uniform
    # ~0.5 % round trip owned by the cumulus caller fixes F14/F18), never a column-selective limit:
    ratio = got["qfx"] / np.asarray(QFX_COLS, np.float64)
    np.testing.assert_allclose(ratio[BEYOND], ratio[~BEYOND].mean(), rtol=1e-6)


def test_mynn_view_alone_applies_the_driver_limits(monkeypatch):
    from gpuwrf.coupling import physics_couplers as C

    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")  # release flags for producer AND consumer
    monkeypatch.setenv("GPUWRF_MYNN_PHY_EXNER", "1")
    state, _ = _extreme_noahmp_step(monkeypatch)
    grid = smoke._land_grid()
    # Op by op, like the eager reference below: a compiled program may let XLA:CPU reassociate the phy_prep
    # divides (E130/E186) -> <= 1 REAL4 ulp vs eager on the beyond columns.
    with jax.disable_jit():
        view = C._surface_fluxes_from_state(state, grid)
    f32 = lambda a: np.asarray(jnp.asarray(a, jnp.float32))  # noqa: E731
    shared = [f32(state.theta_flux), f32(state.qv_flux), f32(state.fltv)]
    got = [np.asarray(view.theta_flux), np.asarray(view.qv_flux), np.asarray(view.fltv)]
    for a in shared + got:  # E200: a degenerate operand (rho 0 -> inf == inf) must not pass vacuously
        assert np.isfinite(a).all()
    for g, h in zip(got, shared):
        assert g.dtype == np.float32 and np.array_equal(_bits(g[~BEYOND]), _bits(h[~BEYOND]))
        assert not np.any(_bits(g[BEYOND]) == _bits(h[BEYOND]))
    # Beyond the limits: the oracle-proven driver statements on independently limited REAL4 grid HFX/QFX.
    column = C._surface_column_view(state, grid)
    hfx1 = np.clip(f32(state.hfx), -600.0, 1200.0)
    qfx1 = np.clip(f32(state.qfx), -3e-4, 9e-4)
    expect = mynn_driver_surface_fluxes(
        jnp.asarray(hfx1), jnp.asarray(qfx1), jnp.asarray(column.rho[..., 0], jnp.float32),
        jnp.maximum(jnp.asarray(column.qv[..., 0], jnp.float32), 0.0), jnp.asarray(state.t_skin, jnp.float32),
        jnp.asarray(column.p[..., 0], jnp.float32), exner=jnp.asarray(column.exner[..., 0], jnp.float32))
    for g, e in zip(got, (expect.flt, expect.flqv, expect.fltv)):
        assert np.isfinite(np.asarray(e)).all()
        assert np.array_equal(_bits(g[BEYOND]), _bits(np.asarray(e)[BEYOND]))
    # The re-formation uses the producer's own operands: inside the limits it reproduces the shared handles
    # (to <= 2 REAL4 ulp here: this legacy smoke State carries f64 qv, the producer's sqv then rounds once more).
    for e, h in zip((expect.flt, expect.flqv, expect.fltv), shared):
        e, h = np.asarray(e)[~BEYOND], h[~BEYOND]
        assert np.all(np.abs(e - h) <= 2 * np.spacing(np.abs(h))), (e, h)
    for name in ("ustar", "tau_u", "tau_v", "rhosfc", "xland", "t_skin"):
        np.testing.assert_array_equal(np.asarray(getattr(view, name)), f32(getattr(state, name)))


def test_inside_the_limits_the_mynn_view_returns_the_shared_handles(monkeypatch):
    """No column beyond the limits (PROD/WN3 always, MY10): the select passes the shared handles, BYTE."""
    from gpuwrf.coupling import physics_couplers as C

    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    monkeypatch.setenv("GPUWRF_MYNN_PHY_EXNER", "1")
    monkeypatch.setenv("GPUWRF_MYNN_FLTV_WRF", "1")
    state, _ = _noahmp_step(grid_backed=True)
    hfx, qfx = np.asarray(state.hfx, np.float32), np.asarray(state.qfx, np.float32)
    assert np.all((hfx >= -600) & (hfx <= 1200) & (qfx >= -3e-4) & (qfx <= 9e-4)) and np.any(hfx != 0)
    view = C._surface_fluxes_from_state(state, smoke._land_grid())
    for got, shared in ((view.theta_flux, state.theta_flux), (view.qv_flux, state.qv_flux), (view.fltv, state.fltv)):
        got, shared = np.asarray(got), np.asarray(jnp.asarray(shared, jnp.float32))
        assert np.isfinite(got).all() and np.array_equal(_bits(got), _bits(shared))


def test_limit_view_is_branch_free_and_forms_kts_operands_only(monkeypatch):
    """Host-transfer gate + E41 PROD d01. No control flow: a cond (or while) predicate is read on the host every
    step inside the stepping while loops (C5: +1 DtoH + sync per d02 step). And no level-kts operand may come from a
    3-D intermediate (full-column phy_prep rho/Exner, the layout-pinned column view): read in the step graph, those
    tipped PROD d01 into oversized multi-phase kInput transpose fusions. Every re-formation op is (ny, nx); only the
    slices of the State's native fields reach two levels (ph faces kts/kts+1)."""
    from gpuwrf.coupling import physics_couplers as C

    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    monkeypatch.setenv("GPUWRF_MYNN_PHY_EXNER", "1")
    state, _ = _extreme_noahmp_step(monkeypatch)
    grid = smoke._land_grid()
    handles = tuple(jnp.asarray(f, jnp.float32) for f in (state.theta_flux, state.qv_flux, state.fltv))
    jaxpr = jax.make_jaxpr(lambda st, *h: C._mynn_driver_limited_handles(st, grid, *h))(state, *handles)  # E140

    def eqns(j):  # every equation, nested jit bodies included (jnp.where is a pjit)
        for e in j.eqns:
            yield e
            for v in e.params.values():
                for sub in (v if isinstance(v, (tuple, list)) else (v,)):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        yield from eqns(sub)

    every = list(eqns(jaxpr.jaxpr))
    names = {e.primitive.name for e in every}
    assert not {"cond", "while", "transpose"} & names, names
    assert {"pow", "div", "log", "select_n"} <= names, names  # the re-formation is really traced (E114)
    ncol = int(np.prod(handles[0].shape))
    slicing = {"slice", "dynamic_slice", "squeeze", "gather", "convert_element_type"}
    sizes = [(e.primitive.name, v.aval.shape) for e in every for v in e.outvars
             if e.primitive.name not in {"pjit", "closed_call"}
             and int(np.prod(v.aval.shape)) > (2 if e.primitive.name in slicing else 1) * ncol]
    assert not sizes, sizes


_PRODUCERS = ("physics/noahmp_coupler.py", "coupling/lake_surface_hook.py", "coupling/ruc_surface_hook.py",
              "physics/surface_layer.py")


def _driver_calls(path):
    """Driver calls per module-level function (a call in a nested helper counts once, for its outer function)."""
    tree = ast.parse((ROOT / "src/gpuwrf" / path).read_text())
    return [(fn.name, call) for fn in tree.body if isinstance(fn, ast.FunctionDef)
            for call in ast.walk(fn) if isinstance(call, ast.Call)
            and getattr(call.func, "id", None) == "mynn_driver_surface_fluxes"]


@pytest.mark.parametrize("path", _PRODUCERS)
def test_shared_handle_producers_stay_unlimited(path):
    calls = _driver_calls(path)
    assert len(calls) == 1, path
    assert all(k.arg != "pbl_limits" for k in calls[0][1].keywords), path


def test_only_the_mynn_consumer_view_limits():
    calls = _driver_calls("coupling/physics_couplers.py")
    assert [name for name, _ in calls] == ["_mynn_driver_limited_handles"]
    kw = {k.arg: k.value for k in calls[0][1].keywords if k.arg}
    assert isinstance(kw.get("pbl_limits"), ast.Constant) and kw["pbl_limits"].value is True


# --- 2-step operational physics block ------------------------------------------------------------------
def test_two_step_operational_mynn_reads_the_current_bottom_faces(monkeypatch):
    from gpuwrf.coupling import physics_couplers as C
    from gpuwrf.runtime.operational_state import initial_operational_carry

    seen = []
    real = C.step_mynn_pbl_column_with_pblh

    def spy(column, *args, **kwargs):
        seen.append(np.asarray(column.w))
        return real(column, *args, **kwargs)

    monkeypatch.setattr(C, "step_mynn_pbl_column_with_pblh", spy)
    grid = smoke._grid()
    nml = smoke._namelist(grid, mp_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0, use_noahmp=False)
    smoke._resolve_operational_suite(nml)
    state = smoke._base_state(grid).replace(w=_terrain_w(grid, seed=1))
    carry = initial_operational_carry(state)
    entries = []
    for step in (1, 2):
        entries.append(np.asarray(carry.state.w))
        forcing = smoke._physics_step_forcing(carry, nml, (step - 1) * nml.dt_s, run_radiation=False,
                                              first_timestep=step == 1)
        assert smoke._all_finite(forcing.state)
        # the dycore moves w between steps: the next MYNN call must read the new faces, not a held copy
        carry = forcing.carry.replace(state=forcing.state.replace(w=_terrain_w(grid, seed=2)))
    assert len(seen) == 2
    for w_entry, w_kernel in zip(entries, seen):
        expect = np.moveaxis(w_entry[:-1], 0, -1).reshape(grid.ny * grid.nx, grid.nz)
        np.testing.assert_array_equal(w_kernel.reshape(expect.shape), expect.astype(w_kernel.dtype))
    assert not np.array_equal(seen[0], seen[1])
