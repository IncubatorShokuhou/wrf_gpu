"""topo_wind 1/2 (WRF 4.7.1 start_em CTOPO/CTOPO2 -> YSU/Shin-Hong), lane w41-topowind.

Truth: pristine WRF 4.7.1 (never the port itself, E98).
* start_em.F:1535-1628 compiled verbatim (proofs/w41_topowind/build_ctopo_oracle.py): real WN3 d03
  (1 km Tenerife) + Swiss RD11 (3 km Alps) HGT/XLAND/VAR, VAR_SSO from a lane geogrid run with
  the NCAR varsso_10m dataset, plus a sigma-small variant so every branch is active.
* the unmodified YSU public call (t-ysuqc oracle) on 147 real CPU-WRF columns (Swiss c02/c07, c40,
  c85, WN3 0227 d02/d03) carrying the CTOPO/CTOPO2 start_em gives their cells
  (proofs/w41_topowind/build_ysu_ctopo_oracle.py); Shin-Hong likewise on 80 c07 rows.
"""
import ctypes
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/w41_topowind"
EPS32 = float(np.finfo(np.float32).eps)
Q_SUBNORMAL_FLOOR = 1.0e-30
FIELDS = (("u", 0), ("v", 1), ("theta", 2), ("qv", 3), ("qc", 4), ("qi", 5), ("exch_h", 6), ("exch_m", 7))
SCALARS = (("pblh", 0), ("kpbl", 1), ("wstar", 2), ("delta", 3), ("u10", 4), ("v10", 5))


# ---------------------------------------------------------------- start_em CTOPO/CTOPO2

CTOPO_CASES = [(case, key) for case, keys in (
    ("wn3_d03", ("tw1", "tw1_periodic", "tw2", "tw2_periodic")),
    ("swiss_d01", ("tw1", "tw2")), ("wn3_d03_sigma_small", ("tw1",))) for key in keys]


@pytest.mark.parametrize("case,key", CTOPO_CASES)
def test_start_em_ctopo_is_byte_identical_to_pristine_wrf(case, key):
    from gpuwrf.io.topo_wind_static import start_em_ctopo
    d = np.load(FIX / "ctopo_oracle.npz")
    periodic = key.endswith("periodic")
    ctopo, ctopo2 = start_em_ctopo(int(key[2]), d[f"{case}_ht"], d[f"{case}_xland"], var_sso=d[f"{case}_var_sso"],
                                   var2d=d[f"{case}_var2d"], periodic_x=periodic, periodic_y=periodic)
    assert ctopo.dtype == ctopo2.dtype == np.float32
    np.testing.assert_array_equal(ctopo.view(np.uint32), d[f"{case}_{key}_ctopo"].view(np.uint32))
    np.testing.assert_array_equal(ctopo2.view(np.uint32), d[f"{case}_{key}_ctopo2"].view(np.uint32))


def test_ctopo_oracle_census_covers_every_start_em_branch():
    """E154: land log / land sigma<=e / water, all four lap_hgt regimes, vfac capped+uncapped."""
    report = json.loads((ROOT / "proofs/w41_topowind/CTOPO_ORACLE.json").read_text())
    assert report["PASS_byte_exact_wrf_and_O0"] and report["variants"]["wrf_fcoptim"]["fma_instructions"] == 0
    runs = [r["census"] | {"key": k} for c in report["cases"].values() for k, r in c["runs"].items()]
    tw1 = [r for r in runs if r["key"].startswith("tw1")]
    tw2 = [r for r in runs if r["key"].startswith("tw2")]
    for branch in ("land_log", "land_sigma_le_e", "water", "lap_blend", "lap_ramp", "lap_zero"):
        assert max(r[branch] for r in tw1) > 0, branch
    for branch in ("land_capped", "land_uncapped"):
        assert max(r[branch] for r in tw2) > 0, branch


def test_topo_wind_admission_refuses_like_wrf(tmp_path):
    from gpuwrf.io.topo_wind_static import load_topo_wind_static, topo_wind_inputs

    class Run:
        def __init__(self, fields):
            self.fields = fields

        def load_wrfinput(self, domain, name, lazy=True):
            if name not in self.fields:
                raise KeyError(name)
            return self.fields[name]

    zero = np.zeros((4, 5), np.float32)
    with pytest.raises(ValueError, match="topo_wind requires VAR_SSO"):
        topo_wind_inputs(Run({"VAR_SSO": zero, "GOT_VAR_SSO": np.array([0])}), "d01", 1)
    with pytest.raises(ValueError, match="no VAR_SSO"):
        topo_wind_inputs(Run({}), "d01", 1)
    with pytest.raises(ValueError, match="geogrid VAR"):
        topo_wind_inputs(Run({}), "d01", 2)
    with pytest.raises(ValueError, match="not a WRF 4.7.1 option"):
        topo_wind_inputs(Run({}), "d01", 3)
    got = topo_wind_inputs(Run({"VAR_SSO": zero + 50.0, "GOT_VAR_SSO": np.array([1])}), "d01", 1)
    assert set(got) == {"var_sso"}

    class Grid:
        terrain_height = zero + 100.0
    # WRF computes CTOPO for every PBL but only YSU/Shin-Hong read it: MYNN is a warned no-op,
    # and topo_wind=1 without VAR_SSO is refused for MYNN too (start_em.F:222 is PBL-independent).
    with pytest.warns(UserWarning, match="no effect with bl_pbl_physics=5"):
        assert load_topo_wind_static(Run({"VAR": zero + 300.0}), "d01", topo_wind=2, bl_pbl_physics=5,
                                     grid=Grid, xland=zero + 1.0) is None
    with pytest.raises(ValueError, match="VAR_SSO"):
        load_topo_wind_static(Run({"VAR_SSO": zero, "GOT_VAR_SSO": np.array([0])}), "d01", topo_wind=1,
                              bl_pbl_physics=5, grid=Grid, xland=zero + 1.0)
    static = load_topo_wind_static(Run({"VAR": zero + 300.0}), "d01", topo_wind=2, bl_pbl_physics=1,
                                   grid=Grid, xland=zero + 1.0)
    assert static.ctopo2 is None and float(static.ctopo[0, 0]) == pytest.approx(1.575 ** 2)


# ---------------------------------------------------------------- YSU with real CTOPO/CTOPO2

def _ysu_fixture():
    meta = json.loads((FIX / "ysu_ctopo_oracle.json").read_text())
    path = FIX / "ysu_ctopo_oracle.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["fixture_sha256"]
    assert meta["reference_build"]["fma_instructions"] == 0
    return np.load(path), meta


def _ysu_groups(d):
    keys = {}
    for i in range(len(d["nz"])):
        keys.setdefault((int(d["topdown"][i]), bool(d["flag_qi"][i]), float(d["surface"][i, 16])), []).append(i)
    return keys.items()


def _ysu_run(d, idxs, flag_qi, topdown, eager):
    from gpuwrf.physics.pbl_ysu import ysu_columns
    n = int(d["nz"][idxs[0]])
    prof, surf = d["profiles"][idxs], d["surface"][idxs]
    with jax.enable_x64(False):
        real = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        c = lambda j: real(prof[:, :n, j])  # noqa: E731
        s = lambda j: real(surf[:, j])  # noqa: E731
        kw = dict(psfc=s(0), znt=s(1), ust=s(2), psim=s(3), psih=s(4), xland=s(5), hfx=s(6), qfx=s(7), wspd=s(8),
                  br=s(9), u10=s(10), v10=s(11), uoce=s(12), voce=s(13), ctopo=s(14), ctopo2=s(15),
                  dt=float(surf[0, 16]), qc=c(4), qi=c(5) if flag_qi else None, rthraten=c(10),
                  ysu_topdown_pblmix=topdown)
        args = (c(0), c(1), c(2), c(3), c(6), real(prof[:, :, 7]), c(8), c(9))
        call = lambda *a: ysu_columns(*a, **kw)  # noqa: E731
        if eager:
            with jax.disable_jit():
                out = call(*args)
        else:
            out = jax.jit(call)(*args)
    return n, {k: np.asarray(v) for k, v in out.items()}


@pytest.fixture
def glibc_transcendentals(monkeypatch):
    """exp/tanh through glibc expf/tanhf, as gfortran -O0 calls them (t-ysuqc rule)."""
    import gpuwrf.physics.pbl_ysu as ysu
    libm = ctypes.CDLL("libm.so.6")
    for name in ("expf", "tanhf"):
        getattr(libm, name).restype = ctypes.c_float
        getattr(libm, name).argtypes = [ctypes.c_float]

    def host(fn):
        vec = np.vectorize(lambda x: getattr(libm, fn)(float(x)), otypes=[np.float32])
        return lambda x: jax.pure_callback(lambda a: vec(np.asarray(a)).astype(np.float32),
                                           jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

    class GlibcJnp:
        exp = staticmethod(host("expf"))
        tanh = staticmethod(host("tanhf"))

        def __getattr__(self, name):
            return getattr(jnp, name)

    monkeypatch.setattr(ysu, "jnp", GlibcJnp())


def test_literal_real4_ysu_with_real_ctopo_is_byte_identical_to_pristine_wrf(glibc_transcendentals):
    d, _ = _ysu_fixture()
    checked = 0
    for (topdown, flag_qi, _dt), idxs in _ysu_groups(d):
        n, out = _ysu_run(d, idxs, flag_qi, topdown, eager=True)
        for pos, i in enumerate(idxs):
            for name, j in FIELDS:
                if name == "qi" and not flag_qi:
                    continue
                wrf, act = d["wrf"][i][:n, j], out[name][pos].astype(np.float32)
                assert np.isfinite(act).all(), (i, name)
                if name in ("qc", "qi"):
                    tiny = np.abs(wrf) < Q_SUBNORMAL_FLOOR
                    assert np.all(np.abs(act[tiny] - wrf[tiny]) <= Q_SUBNORMAL_FLOOR), (i, name)
                    wrf, act = wrf[~tiny], act[~tiny]
                assert np.array_equal(act, wrf), (i, name, np.max(np.abs(act - wrf)))
                checked += 1
            for name, j in SCALARS:
                assert np.float32(out[name][pos]) == d["wrf"][i][j, 8], (i, name)
    assert checked > 1500


def test_compiled_ysu_with_real_ctopo_inside_wrf_input_ulp_envelope():
    d, _ = _ysu_fixture()
    for (topdown, flag_qi, _dt), idxs in _ysu_groups(d):
        n, out = _ysu_run(d, idxs, flag_qi, topdown, eager=False)
        for pos, i in enumerate(idxs):
            for name, j in FIELDS + (("u10", 4), ("v10", 5)):
                if name == "qi" and not flag_qi:
                    continue
                scalar = name in ("u10", "v10")
                lo = d["lower"][i][j, 8] if scalar else d["lower"][i][:n, j]
                hi = d["upper"][i][j, 8] if scalar else d["upper"][i][:n, j]
                act = out[name][pos]
                widen = 2.0 * EPS32 * np.maximum(np.abs(lo), np.abs(hi)) + (Q_SUBNORMAL_FLOOR if name in ("qc", "qi") else 0)
                bad = (act < lo - widen) | (act > hi + widen) | ~np.isfinite(act)
                assert not np.any(bad), (i, name)
            assert int(out["kpbl"][pos]) == int(d["wrf"][i][1, 8]), i


def test_ysu_ctopo_oracle_is_drag_and_blend_sensitive():
    """The columns exercise CTOPO in the momentum diagonal and CTOPO2 in U10/V10 (E39/E154)."""
    d, meta = _ysu_fixture()
    c = meta["census"]
    assert c["ctopo_gt1"] >= 20 and c["ctopo_zero"] >= 4 and c["ctopo2_partial"] >= 2 and c["ctopo2_zero"] >= 4
    assert c["topo_wind1"] >= 20 and c["topo_wind2"] >= 20 and c["u10_blended"] >= 10 and c["wn3_columns"] >= 10
    # Same columns with CTOPO = CTOPO2 = 1 (topo_wind=0) give different RUBLTEN in most rows.
    neutral = {k: d[k] for k in d.files}
    neutral["surface"] = d["surface"].copy()
    neutral["surface"][:, 14:16] = 1.0
    differ = 0
    for (topdown, flag_qi, _dt), idxs in _ysu_groups(d):
        n, out = _ysu_run(neutral, idxs, flag_qi, topdown, eager=False)
        differ += sum(int(np.any(out["u"][pos] != d["wrf"][i][:n, 0])) for pos, i in enumerate(idxs))
    assert differ >= 100


# ---------------------------------------------------------------- real call path

def _ops_namelist(**changes):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_v013_operational_smoke import _grid, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    return grid, _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=changes.pop("bl", 1),
                           sf_sfclay_physics=1, sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0,
                           ra_sw_physics=0)


@pytest.mark.parametrize("bl", (1, 11))
def test_operational_step_hands_start_em_ctopo_to_ysu_and_shinhong(monkeypatch, bl):
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    sys.path.insert(0, str(ROOT / "tests"))
    from test_v013_operational_smoke import _base_state
    grid, namelist = _ops_namelist(bl=bl)
    ctopo = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 4.0
    ctopo2 = jnp.where(ctopo > 1.0, 1.0, 0.5).astype(jnp.float32)
    seen = {}

    def pbl(st, dt, grid, **kwargs):
        seen.update(kwargs)
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        rates = {key: zero for key in ("u", "v", "theta", "qv", "qc", "qi")}
        rates.update(u10=jnp.full((9,), 7.0, jnp.float32), v10=jnp.full((9,), -3.0, jnp.float32))
        return (st, rates) if kwargs.get("_with_sources") or "surface_forcing" in kwargs else st

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, bl, pbl)
    carry = initial_operational_carry(_base_state(grid))
    if carry.history_diagnostics is None:
        from gpuwrf.runtime.operational_state import seed_history_diagnostics
        carry = carry.replace(history_diagnostics=seed_history_diagnostics(carry.state))
    results = {}
    for static, publish in ((TopoWindStatic(ctopo, None), False), (TopoWindStatic(ctopo, ctopo2), True)):
        seen.clear()
        nl = dataclasses.replace(namelist, topo_wind_static=static)
        out = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
        assert "topo_wind_ctopo" in seen, f"bl={bl}: PBL adapter never received the start_em CTOPO"
        np.testing.assert_array_equal(np.asarray(seen["topo_wind_ctopo"][0]), np.asarray(ctopo))
        assert (seen["topo_wind_ctopo"][1] is None) == (not publish)
        results[publish] = out.carry.history_diagnostics
    surface_u10, surface_v10 = np.asarray(results[False].u10), np.asarray(results[False].v10)
    blend = np.asarray(ctopo2) != 1.0
    assert blend.any() and (~blend).any() and not np.any(surface_u10[blend] == 7.0)
    np.testing.assert_array_equal(np.asarray(results[True].u10), np.where(blend, 7.0, surface_u10).astype(surface_u10.dtype))
    np.testing.assert_array_equal(np.asarray(results[True].v10), np.where(blend, -3.0, surface_v10).astype(surface_v10.dtype))
    seen.clear()
    om._physics_step_forcing(carry, namelist, 0., run_radiation=False, first_timestep=True)
    assert "topo_wind_ctopo" not in seen


def test_two_way_feedback_selects_own_ctopo_only_on_the_first_step():
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    from gpuwrf.runtime import operational_mode as om
    grid, namelist = _ops_namelist()
    own, fed = jnp.ones((3, 3), jnp.float32), jnp.full((3, 3), 2.0, jnp.float32)
    nl = dataclasses.replace(namelist, topo_wind_static=TopoWindStatic(fed, fed * 0.5, own, own * 0.25))
    first = om._topo_wind_ctopo(nl, 0.0)
    later = om._topo_wind_ctopo(nl, float(nl.dt_s))
    assert float(first[0][0, 0]) == 1.0 and float(first[1][0, 0]) == 0.25
    assert float(later[0][0, 0]) == 2.0 and float(later[1][0, 0]) == 1.0


def test_ysu_adapter_passes_flat_real_ctopo_columns(monkeypatch):
    from gpuwrf.coupling import scan_adapters as adapters
    from gpuwrf.physics.pbl_ysu import ysu_columns
    sys.path.insert(0, str(ROOT / "tests"))
    from test_ysu_surface_contract import column_state
    state, _, forcing = column_state()
    ny, nx = state.theta.shape[1:]
    ctopo = jnp.full((ny, nx), 3.5, jnp.float32)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return ysu_columns(*args, **kwargs)

    monkeypatch.setattr(adapters, "ysu_columns", spy)
    _, rates = adapters.ysu_pbl_adapter(state, 60., surface_forcing=forcing, _with_sources=True,
                                        topo_wind_ctopo=(ctopo, ctopo * 0.0))
    assert "ctopo" in seen and "ctopo2" in seen, "YSU column never received CTOPO/CTOPO2"
    assert seen["ctopo"].dtype == jnp.float32 and seen["ctopo"].shape == (ny * nx,)
    np.testing.assert_array_equal(np.asarray(seen["ctopo2"]), np.zeros(ny * nx, np.float32))
    # CTOPO2 = 0 makes U10 the lowest-level wind (bl_ysu.F90:1406).
    np.testing.assert_array_equal(np.asarray(rates["u10"]), np.asarray(forcing_u(state, forcing)))


def forcing_u(state, forcing):
    from gpuwrf.coupling.scan_adapters import _pbl_surface_forcing
    f = _pbl_surface_forcing(state, None, surface_forcing=forcing, use_resistances=True)
    return np.asarray(f["u_cols"], np.float32)[:, 0]


def test_shinhong_ctopo2_blend_and_wrf_drag_order():
    """module_bl_shinhong.F:1387 ctopo*ust**2 and :1471-1472 the CTOPO2 U10/V10 blend."""
    from gpuwrf.physics.bl_shinhong import shinhong_columns
    nz, ncol = 10, 3
    z = jnp.linspace(0.0, 1.0, nz, dtype=jnp.float32)
    prof = lambda a, b: jnp.broadcast_to(a + b * z, (ncol, nz)).astype(jnp.float32)  # noqa: E731
    p = prof(95000.0, -20000.0)
    pint = jnp.concatenate([p[:, :1] + 300.0, p - 300.0], axis=1)
    kw = dict(psfc=jnp.full((ncol,), 95300.0, jnp.float32), znt=jnp.full((ncol,), 0.1, jnp.float32),
              ust=jnp.full((ncol,), 0.37, jnp.float32), hfx=jnp.full((ncol,), 40.0, jnp.float32),
              qfx=jnp.full((ncol,), 1e-5, jnp.float32), wspd=jnp.full((ncol,), 6.0, jnp.float32),
              br=jnp.full((ncol,), 0.05, jnp.float32), psim=jnp.full((ncol,), 4.0, jnp.float32),
              psih=jnp.full((ncol,), 5.0, jnp.float32), dt=60.0, xland=jnp.ones((ncol,), jnp.float32),
              u10=jnp.full((ncol,), 4.0, jnp.float32), v10=jnp.full((ncol,), 1.0, jnp.float32), dx=3000.0, dy=3000.0)
    args = (prof(5.0, 10.0), prof(1.0, 2.0), prof(290.0, -40.0), prof(8e-3, -6e-3), p, pint,
            (p / 1e5) ** 0.2857, prof(60.0, 300.0), prof(0.5, -0.4))
    with jax.enable_x64(False):
        base = shinhong_columns(*args, **kw)
        topo = shinhong_columns(*args, **kw, ctopo=jnp.array([1.0, 3.0, 0.0], jnp.float32),
                                ctopo2=jnp.array([1.0, 0.5, 0.0], jnp.float32))
    assert "u10" not in base
    assert np.array_equal(np.asarray(topo["u"])[0], np.asarray(base["u"])[0])        # CTOPO = 1 column unchanged
    assert not np.array_equal(np.asarray(topo["u"])[1], np.asarray(base["u"])[1])    # drag scaled
    c2, u0 = np.array([1.0, 0.5, 0.0], np.float32), np.asarray(args[0])[:, 0]
    np.testing.assert_array_equal(np.asarray(topo["u10"]), (c2 * np.float32(4.0) + (np.float32(1) - c2) * u0))


# ---------------------------------------------------------------- binding, catalog, history

def test_topo_wind_is_honoured_and_cataloged():
    from gpuwrf.io.namelist_binding import namelist_honour_report
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_control
    for tw in (1, 2):
        errors, _ = namelist_honour_report({"physics": {"bl_pbl_physics": 1, "topo_wind": tw}}, ("d01",))
        assert not any("topo_wind" in e for e in errors), errors
        support = classify_control("topo_wind", tw)
        assert support.status is SupportStatus.IMPLEMENTED
        # Notes state exactly the recorded GPU qualification (test_v034_implemented_notes convention).
        reason = support.reason
        assert "GPU-qualified vs CPU-WRF 24 h" in reason and "E41 robustness flag" in reason
        assert "proofs/w41_topowind/GPU_ARMS.json" in reason and "qualification pending" not in reason
    arms = json.loads((ROOT / "proofs/w41_topowind/GPU_ARMS.json").read_text())["arms"]
    for case, domains in (("c02tw1", ("d01",)), ("c07tw1", ("d01",)), ("wn3tw", ("d01", "d02", "d03"))):
        assert arms[case]["rc"] == 0 and arms[case]["platform"] == "gpu" and arms[case]["hours"] == 24
        for d in domains:
            d6 = arms[case]["d6"][d]
            assert d6["verdict"] == "PASS" and d6["paired_frames"] == 25 and d6["coverage_complete"]
        assert arms[case]["e41"]["gate"] == "FAIL"          # disclosed in both notes
    assert "only on WN3 0227 d01" in classify_control("topo_wind", 2).reason
    assert "STACK 400 B" in classify_control("topo_wind", 1).reason
    assert classify_control("topo_wind", 3).status is not SupportStatus.IMPLEMENTED


def test_default_namelist_treedef_unchanged_and_static_is_a_traced_child():
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    _, namelist = _ops_namelist()
    leaves, tree = jax.tree_util.tree_flatten(namelist)
    static = TopoWindStatic(jnp.ones((3, 3), jnp.float32), None)
    leaves_on, tree_on = jax.tree_util.tree_flatten(dataclasses.replace(namelist, topo_wind_static=static))
    assert tree_on != tree and len(leaves_on) == len(leaves) + 1
    back = jax.tree_util.tree_unflatten(tree_on, leaves_on)
    assert isinstance(back.topo_wind_static, TopoWindStatic) and back.topo_wind_static.ctopo2 is None
    # Same structure, other case values -> same treedef (B41: no per-case recompile).
    other = TopoWindStatic(jnp.full((3, 3), 2.0, jnp.float32), None)
    assert jax.tree_util.tree_structure(dataclasses.replace(namelist, topo_wind_static=other)) == tree_on


@pytest.mark.parametrize("tw,present", ((0, False), (1, True), (2, False)))
def test_history_writes_topo_wind_attribute_only_for_option_1(tmp_path, tw, present):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
    path = tmp_path / "wrfinput_d01"
    with Dataset(path, "w") as ds:
        ds.setncattr("DIFF_6TH_FACTOR", np.float32(0.12))
        ds.setncattr("TOPO_WIND", np.int32(1))      # a real.exe run with topo_wind=1
        ds.setncattr("GRID_FDDA", np.int32(0))
    attrs = wrf_history_global_attributes(path, {"physics": {"topo_wind": [tw]}}, "d01", 18.0)
    assert ("TOPO_WIND" in attrs) == present
    if present:
        names = list(attrs)
        assert names[names.index("DIFF_6TH_FACTOR") + 1] == "TOPO_WIND" and attrs["TOPO_WIND"] == 1


def test_two_way_feedback_composes_child_ctopo_onto_parent_like_copy_fcn():
    """Registry ctopo/ctopo2 'u' (copy_fcn): parent over the nest = child average after step 1;
    grandchild values reach the root through the middle domain (deepest edge first)."""
    from gpuwrf.coupling.boundary_feedback import apply_feedback, build_feedback_weights
    from gpuwrf.io.topo_wind_static import feedback_topo_wind_statics
    w12 = build_feedback_weights(parent_grid_ratio=3, i_parent_start=4, j_parent_start=4,
                                 parent_we=12, parent_sn=12, child_we=15, child_sn=15)
    w23 = build_feedback_weights(parent_grid_ratio=3, i_parent_start=3, j_parent_start=3,
                                 parent_we=15, parent_sn=15, child_we=12, child_sn=12)
    ones = lambda n: np.ones((n, n), np.float32)  # noqa: E731
    rng = np.random.default_rng(7)
    own = {"d01": (ones(12), ones(12)),                                  # topo_wind=0 root, YSU
           "d02": (rng.uniform(0.5, 4.0, (15, 15)).astype(np.float32), ones(15)),
           "d03": (rng.uniform(0.0, 6.0, (12, 12)).astype(np.float32),
                   rng.uniform(0.0, 1.0, (12, 12)).astype(np.float32))}
    statics = feedback_topo_wind_statics(own, {"d01": 1, "d02": 1, "d03": 5},
                                         [("d02", "d03"), ("d01", "d02")], {("d02", "d03"): w23, ("d01", "d02"): w12})
    assert statics["d03"] is None                                         # MYNN reads nothing
    fed2 = tuple(np.asarray(apply_feedback(jnp.asarray(own["d02"][k]), jnp.asarray(own["d03"][k]), w23)) for k in (0, 1))
    fed1 = tuple(np.asarray(apply_feedback(jnp.asarray(own["d01"][k]), jnp.asarray(fed2[k]), w12)) for k in (0, 1))
    for d, fed in (("d02", fed2), ("d01", fed1)):
        s = statics[d]
        np.testing.assert_array_equal(np.asarray(s.ctopo), fed[0])
        np.testing.assert_array_equal(np.asarray(s.ctopo2), fed[1])
        np.testing.assert_array_equal(np.asarray(s.ctopo_first), own[d][0])
        np.testing.assert_array_equal(np.asarray(s.ctopo2_first), own[d][1])
    # The topo_wind=0 YSU root still sees the nest's CTOPO from its second step (WRF 'u' feedback).
    assert np.any(np.asarray(statics["d01"].ctopo) != 1.0) and np.all(np.asarray(statics["d01"].ctopo_first) == 1.0)


# ---------------------------------------------------------------- Shin-Hong with real CTOPO/CTOPO2

SH_FIELDS = (("tke", 0), ("el_pbl", 1), ("exch_h", 2), ("u", 3), ("v", 4), ("theta", 5), ("qv", 6), ("qc", 7), ("qi", 8))
SH_SURF = ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "dt", "xland", "u10", "v10", "dx", "dy",
           "corf", "ctopo", "ctopo2")


def _sh_fixture():
    meta = json.loads((FIX / "shinhong_ctopo_oracle.json").read_text())
    path = FIX / "shinhong_ctopo_oracle.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["fixture_sha256"]
    assert meta["reference_build"]["fma_instructions"] == 0
    return np.load(path), meta


def _sh_run(d, idxs, eager):
    from gpuwrf.physics.bl_shinhong import shinhong_columns
    n = int(d["nz"][idxs[0]])
    a, s = d["profiles"][idxs], d["surface"][idxs]
    with jax.enable_x64(False):
        real = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        kw = {name: real(s[:, i]) for i, name in enumerate(SH_SURF) if name not in ("dt", "dx", "dy")}
        kw.update(dt=real(s[0, 9]), dx=real(s[0, 13]), dy=real(s[0, 14]), qc=real(a[:, :n, 9]),
                  qi=real(a[:, :n, 10]), shinhong_tke_diag=jnp.asarray(d["mode"][idxs], jnp.int32))
        args = [real(a[:, :, j] if j == 5 else a[:, :n, j]) for j in range(9)]
        if eager:
            with jax.disable_jit():
                out = shinhong_columns(*args, **kw)
        else:
            out = jax.jit(shinhong_columns)(*args, **kw)
    return n, {k: np.asarray(v) for k, v in out.items()}


def _sh_groups(d):
    groups = {}
    for i, n in enumerate(d["nz"]):
        groups.setdefault(int(n), []).append(i)
    return groups.values()


def test_literal_real4_shinhong_with_real_ctopo_is_byte_identical_to_pristine_wrf():
    d, meta = _sh_fixture()
    c = meta["census"]
    assert c["ctopo_gt1"] >= 20 and c["ctopo_zero"] >= 4 and c["ctopo2_partial"] >= 4 and c["u10_blended"] >= 10
    for idxs in _sh_groups(d):
        n, out = _sh_run(d, idxs, eager=True)
        for pos, i in enumerate(idxs):
            for name, j in SH_FIELDS:
                wrf, act = d["wrf"][i][:n, j], out[name][pos].astype(np.float32)
                if name in ("qc", "qi"):
                    tiny = np.abs(wrf) < Q_SUBNORMAL_FLOOR
                    assert np.all(np.abs(act[tiny].astype(np.float64) - wrf[tiny]) <= Q_SUBNORMAL_FLOOR), (i, name)
                    wrf, act = wrf[~tiny], act[~tiny]
                assert act.tobytes() == wrf.tobytes(), (i, name, float(np.max(np.abs(act.astype(np.float64) - wrf))))
            for name, j in (("u10", 4), ("v10", 5)):
                assert out[name][pos].astype(np.float32).tobytes() == d["wrf"][i][j, 9].tobytes(), (i, name)


def test_compiled_shinhong_with_real_ctopo_inside_wrf_input_ulp_envelope():
    d, _ = _sh_fixture()
    for idxs in _sh_groups(d):
        n, out = _sh_run(d, idxs, eager=False)
        for pos, i in enumerate(idxs):
            for name, j in SH_FIELDS[2:] + (("u10", None), ("v10", None)):
                if j is None:
                    j10 = 4 if name == "u10" else 5
                    lo, hi = d["lower"][i][j10, 9], d["upper"][i][j10, 9]
                else:
                    lo, hi = d["lower"][i][:n, j], d["upper"][i][:n, j]
                act = np.asarray(out[name][pos], np.float32)
                widen = 2.0 * EPS32 * np.maximum(np.abs(lo), np.abs(hi)) + (Q_SUBNORMAL_FLOOR if name in ("qc", "qi") else 0)
                bad = (act < lo - widen) | (act > hi + widen) | ~np.isfinite(act)
                assert not np.any(bad), (i, name)


def test_legacy_checkpoint_roundtrips_the_static_and_nested_resume_recomputes_it(tmp_path):
    """Registry ctopo/ctopo2 are restart ('r') fields: the pickled namelist keeps them; the
    nested resume rebuilds statics through _load_domains (restart_store hashes the namelist)."""
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    from gpuwrf.runtime.checkpoint import read_checkpoint, write_checkpoint
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    sys.path.insert(0, str(ROOT / "tests"))
    from test_m7_restart_checkpoint_roundtrip import _synthetic_state, _synthetic_tendencies
    grid = GridSpec.canary_3km_template()
    ctopo = jnp.asarray(np.random.default_rng(1).uniform(0, 6, (grid.ny, grid.nx)), jnp.float32)
    static = TopoWindStatic(ctopo, jnp.clip(ctopo / 3.0, 0.0, 1.0), ctopo * 0.5, None)
    namelist = OperationalNamelist(grid=grid, tendencies=_synthetic_tendencies(grid), metrics=grid.metrics,
                                   dt_s=10.0, acoustic_substeps=10, bl_pbl_physics=1, topo_wind_static=static)
    path = tmp_path / "restart.pkl"
    write_checkpoint(_synthetic_state(grid), namelist, grid, 3, path)
    _, restored, _, _ = read_checkpoint(path)
    back = restored.topo_wind_static
    assert isinstance(back, TopoWindStatic) and back.ctopo2_first is None
    for a, b in zip(static, back):
        if a is not None:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    from gpuwrf.runtime import restart_store
    import inspect
    assert "canonical_digest(bundle.namelist)" in inspect.getsource(restart_store.nested_identity)
