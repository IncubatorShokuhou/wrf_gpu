"""sf_urban_physics=1 (SLUCM) operational wiring: binding/refusals, OFF invariance, hook mapping."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "v040" / "urban_slucm" / "savepoints"

from gpuwrf.coupling import urban_slucm_hook as hook  # noqa: E402
from gpuwrf.physics import urban_slucm as us  # noqa: E402


def _nl(urban, lsm, ndom=1):
    return {"physics": {"sf_urban_physics": [urban] * ndom, "sf_surface_physics": [lsm] * ndom},
            "dynamics": {}, "noah_mp": {}}


def test_honour_check_refusals():
    from gpuwrf.io.namelist_binding import namelist_honour_report

    errs, _ = namelist_honour_report(_nl(1, 4), ("d01",))
    assert not [e for e in errs if "sf_urban_physics" in e], errs
    errs, _ = namelist_honour_report(_nl(1, 2), ("d01",))
    assert any("sf_urban_physics=1 with sf_surface_physics=2" in e for e in errs), errs
    errs, _ = namelist_honour_report(_nl(1, 4), ("d01",), replay=True)
    assert any("sf_urban_physics=1" in e and "replays CPU land" in e for e in errs), errs
    errs, _ = namelist_honour_report(_nl(0, 4), ("d01",), replay=True)
    assert not [e for e in errs if "sf_urban_physics" in e]


def test_catalog_and_validation_accept_slucm():
    from gpuwrf.contracts.physics_registry import ACCEPTED_SF_URBAN_PHYSICS
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    assert 1 in ACCEPTED_SF_URBAN_PHYSICS
    s = classify_scheme("sf_urban_physics", 1)
    assert s.status is SupportStatus.IMPLEMENTED
    assert "SLUCM" in s.reason and ("qualification pending" in s.reason or "GPU-qualified vs CPU-WRF" in s.reason)
    for code in (2, 3):
        support=classify_scheme("sf_urban_physics", code)
        assert support.status is SupportStatus.IMPLEMENTED
        # BEP/BEM stay CPU-oracle-qualified with the coupled GPU forecast still open; w41-writergaps F3
        # (5fc019d360) reworded code 3 to "coupled-forecast fidelity vs CPU-WRF pending".
        assert "CPU-oracle-qualified" in support.reason
        assert ("qualification pending" in support.reason
                or "coupled-forecast fidelity vs CPU-WRF pending" in support.reason), support.reason
        assert "GPU-qualified" not in support.reason, support.reason


def _namelist():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import Tendencies
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    grid = GridSpec.canary_3km_template()
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    tend = Tendencies(z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)), z((nz, ny, nx)),
                      z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)))
    return OperationalNamelist.from_grid(grid, tendencies=tend)


def test_namelist_aux_unchanged_when_off_and_roundtrips_bundle():
    import dataclasses

    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    from gpuwrf.runtime.operational_mode import _UrbanBundleHolder

    off = _namelist()
    children, aux = off.tree_flatten()
    assert not any(isinstance(a, _UrbanBundleHolder) for a in aux)
    bundle = hook.UrbanBundle(table=us.parse_urbparm(SAVE / "URBPARM.TBL.default"), isurban=13, natural=14)
    on = dataclasses.replace(off, urban_bundle=bundle, sf_urban_physics=1)
    children_on, aux_on = on.tree_flatten()
    assert len(aux_on) == len(aux) + 1
    # Later optional scheme metadata follows the original urban holder.
    holders = [value for value in aux_on if isinstance(value, _UrbanBundleHolder)]
    assert len(holders) == 1 and holders[0].value is bundle
    back = type(on).tree_unflatten(aux_on, children_on)
    assert back.urban_bundle is bundle and int(back.sf_urban_physics) == 1
    assert type(off).tree_unflatten(aux, children).urban_bundle is None
    assert static_config_hash(off) != static_config_hash(on)
    assert static_config_hash(off) == static_config_hash(dataclasses.replace(on, urban_bundle=None,
                                                                             sf_urban_physics=0))


def test_operational_carry_field_appended_last_and_restart_registered():
    from dataclasses import fields as dc_fields

    from gpuwrf.io import wrfrst_netcdf as rst
    from gpuwrf.runtime.operational_state import OperationalCarry

    names = [f.name for f in dc_fields(OperationalCarry)]
    assert names.index("urban_slucm") < names.index("urban_bep")
    assert names.index("urban_slucm") > names.index("fdda")
    assert "urban_slucm" in rst.OPTIONAL_CARRY_FIELDS and "urban_slucm" in rst._LATE_PYTREE_CARRY_FIELDS
    assert "urban_slucm" not in rst.UNSUPPORTED_CARRY_FIELDS


def test_urban_noahmp_static_remap():
    table = us.parse_urbparm(SAVE / "URBPARM.TBL.default")
    bundle = hook.UrbanBundle(table=table, isurban=13, natural=14)
    ivg = jnp.asarray([[13, 10], [5, 13]], jnp.int32)
    shdmax = jnp.asarray([[0.2, 0.5], [0.7, 0.1]], jnp.float32)

    class _S(SimpleNamespace):
        def replace(self, **u):
            d = dict(self.__dict__)
            d.update(u)
            return _S(**d)

    out = hook.urban_noahmp_static(_S(ivgtyp=ivg, shdmax=shdmax), bundle)
    assert np.array_equal(np.asarray(out.ivgtyp), [[14, 10], [5, 14]])
    assert np.allclose(np.asarray(out.shdmax), [[0.96, 0.5], [0.7, 0.96]])


def test_mptable_natural_modis():
    from gpuwrf.config.paths import wrf_run_dir

    if not (wrf_run_dir() / "MPTABLE.TBL").exists():
        pytest.skip("pristine WRF run/ tables not linked")
    assert hook.mptable_natural(wrf_run_dir()) == 14


def test_hook_forcing_mapping_reproduces_oracle_step1():
    """apply_urban_slucm maps Noah-MP forcing (P_ML, ZLVL, precip rates) exactly like noahmp_urban."""
    z = np.load(SAVE / "slucm_fp64_default.npz")
    table = us.parse_urbparm(SAVE / "URBPARM.TBL.default", dtype=np.float64)
    bundle = hook.UrbanBundle(table=table, isurban=13, natural=14)
    f64 = lambda k: jnp.asarray(z["in_" + k], jnp.float64)  # noqa: E731
    n = len(z["regime_names"])
    tslb = jnp.stack([f64(f"tslb{k}") for k in range(1, 5)], axis=0)
    carry = hook.initial_urban_carry(bundle, ivgtyp=jnp.full((n,), 13, jnp.int32), tsk=f64("tsk"), tslb=tslb,
                                     frc_urb2d=f64("frc_in"), dtype=jnp.float64)
    dt = float(z["in_dt"][0])
    qv = f64("qv1")
    forcing = SimpleNamespace(
        sfctmp=f64("t3d1"), sfcprs=(f64("p8w2") + f64("p8w1")) * 0.5, psfc=f64("p8w1"), uu=f64("u1"),
        vv=f64("v1"), soldn=f64("swdown"), lwdn=f64("glw"), zlvl=0.5 * f64("dz8w1"),
        prcpconv=jnp.zeros(n), prcpnonc=f64("rainbl") / dt)
    from gpuwrf.contracts.noahmp_state import NoahMPFluxes
    zero = jnp.zeros(n)
    nm = NoahMPFluxes(hfx=f64("hfx"), lh=f64("lh"), qfx=f64("qfx"), grdflx=f64("grdflx"), tsk=f64("tsk"),
                      qsfc=f64("qsfc"), znt=zero, emiss=zero, albedo=f64("albedo"), chs=zero)
    nm2, ust2, carry2, step = hook.apply_urban_slucm(
        bundle, carry, forcing=forcing, nm=nm, ust=f64("ust"), chs=f64("chs"), chs2=f64("chs2"),
        cqs2=f64("cqs2"), qv_mixing_ratio=qv, dt=dt)
    for name, got in (("TSK", nm2.tsk), ("HFX", nm2.hfx), ("QFX", nm2.qfx), ("LH", nm2.lh),
                      ("GRDFLX", nm2.grdflx), ("ALBEDO", nm2.albedo), ("QSFC", nm2.qsfc), ("UST", ust2),
                      ("TR", carry2.state.tr), ("TC", carry2.state.tc), ("TGL1", carry2.state.tgl[:, 0])):
        ref = z[name][:, 1]
        assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-9 * max(np.max(np.abs(ref)), 1e-30), name
    t2, q2, u10, v10 = hook.urban_history_overwrite(
        f64("fveg") * f64("t2mv") + (1 - f64("fveg")) * f64("t2mb"),
        f64("fveg") * f64("q2mv") + (1 - f64("fveg")) * f64("q2mb"), f64("u1"), f64("v1"), step, f64("psfc"))
    for name, got in (("T2", t2), ("Q2", q2), ("U10", u10), ("V10", v10)):
        ref = z[name][:, 1]
        assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-9 * max(np.max(np.abs(ref)), 1e-30), name


def test_urban_carry_restart_pytree_roundtrip():
    """wrfrst_netcdf stores pytree groups as a pickled treedef + leaf_NNN arrays (reader :1400)."""
    import base64
    import pickle

    table = us.parse_urbparm(SAVE / "URBPARM.TBL.default")
    bundle = hook.UrbanBundle(table=table, isurban=13, natural=14)
    ivg = jnp.asarray([[13, 7], [13, 13]], jnp.int32)
    tsk = jnp.asarray([[290.0, 291.0], [289.5, 292.0]], jnp.float32)
    tslb = jnp.stack([tsk - k for k in range(4)], axis=0)
    carry = hook.initial_urban_carry(bundle, ivgtyp=ivg, tsk=tsk, tslb=tslb, frc_urb2d=jnp.zeros((2, 2)))
    assert np.asarray(carry.utype).tolist() == [[2, 0], [2, 2]]
    assert np.allclose(np.asarray(carry.frc_urb), [[0.9, 0.0], [0.9, 0.9]])
    leaves, treedef = jax.tree.flatten(carry)
    blob = base64.b64encode(pickle.dumps(treedef))
    back = jax.tree.unflatten(pickle.loads(base64.b64decode(blob)), [np.asarray(x) for x in leaves])
    assert type(back) is hook.UrbanCarry and type(back.state) is us.UrbanState
    for a, b in zip(jax.tree.leaves(back), leaves):
        assert np.array_equal(np.asarray(a), np.asarray(b)) and np.asarray(a).dtype == np.asarray(b).dtype


def test_scan_suite_refuses_slucm_without_noahmp_or_bundle():
    import importlib.util

    from gpuwrf.runtime.operational_mode import UnsupportedSchemeSelection, _resolve_operational_suite

    spec = importlib.util.spec_from_file_location("g3", ROOT / "tests" / "test_v022_g3city_urban_lake.py")
    g3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g3)
    for overrides in ({"sf_urban_physics": 1}, {"sf_urban_physics": 1, "use_noahmp": True, "sf_surface_physics": 4},
                      {"sf_urban_physics": 1, "urban_bundle": object()}):
        overrides = dict(tracer_opt=0, tracer_config=None, windfarm_opt=0,
                         sf_ocean_physics=0, isftcflx=0, lower_boundary=None, **overrides)
        with pytest.raises(UnsupportedSchemeSelection, match="sf_urban_physics=1"):
            _resolve_operational_suite(g3._minimal_operational_namelist(**overrides))


def _adapter_case():
    """7 columns: 4 ISURBAN (real WN3 0227 d03 urban forcing, day x2 + night x2), 2 rural land, 1 water."""
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP

    z = np.load(SAVE / "slucm_fp64_default.npz")
    names = list(z["regime_names"])
    pick = [names.index(n) for n in ("day_j16_i44", "day_j24_i37", "night_j16_i44", "night_j24_i37")]
    src = pick + [pick[0], pick[2], pick[1]]           # rural/water columns reuse real forcing too
    inp = lambda k: np.asarray(z["in_" + k], np.float64)[src]  # noqa: E731
    n = len(src)
    shape = (1, n)
    ivg = np.array([13, 13, 13, 13, 10, 5, 17])
    xland = np.array([1.0] * 6 + [2.0])
    psfc = inp("p8w1")
    dz = inp("dz8w1")
    p1 = psfc - 0.5 * dz * 1.15 * 9.81
    t = inp("t3d1")
    s2 = lambda a: jnp.asarray(np.asarray(a, np.float64).reshape(shape))  # noqa: E731
    col = lambda a: jnp.asarray(np.asarray(a, np.float64).reshape(shape + (1,)))  # noqa: E731

    class _State(SimpleNamespace):
        def replace(self, **u):
            d = dict(self.__dict__)
            d.update(u)
            return _State(**d)

    state = _State(u=col(inp("u1")), v=col(inp("v1")), theta=col(t * (P0_PA / p1) ** R_D_OVER_CP),
                   qv=col(inp("qv1")), p=col(p1), dz=col(dz), t_skin=s2(inp("tsk")), xland=s2(xland),
                   roughness_m=s2(np.where(ivg == 13, 0.8, np.where(xland > 1.5, 0.0015, 0.1))),
                   mavail=s2(np.where(xland > 1.5, 1.0, 0.6)), ustar=s2(inp("ust")))
    zz = jnp.zeros(shape)
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float64), shape)  # noqa: E731
    tslb = np.stack([inp(f"tslb{k}") for k in range(1, 5)], 0).reshape(NSOIL, 1, n)
    land = NoahMPLandState(
        tslb=jnp.asarray(tslb), smois=jnp.full((NSOIL, 1, n), 0.25), sh2o=jnp.full((NSOIL, 1, n), 0.25),
        smcwtd=full(0.25), isnow=jnp.zeros(shape, jnp.int32), tsno=jnp.full((NSNOW, 1, n), 273.0),
        snice=jnp.zeros((NSNOW, 1, n)), snliq=jnp.zeros((NSNOW, 1, n)),
        zsnso=jnp.broadcast_to(jnp.asarray([0.0, 0.0, 0.0, -0.1, -0.4, -1.0, -2.0]).reshape(NSNOW + NSOIL, 1, 1),
                               (NSNOW + NSOIL, 1, n)),
        snowh=zz, sneqv=zz, sneqvo=zz, tauss=zz, albold=full(0.65), tv=s2(inp("tsk")), tg=s2(inp("tsk")),
        tah=s2(inp("tsk")), eah=full(2000.0), canliq=zz, canice=zz, fwet=zz, lai=full(1.5), sai=full(0.3),
        cm=full(0.01), ch=full(0.01), t_skin=s2(inp("tsk")), qsfc=s2(1.02 * inp("qv1")), znt=full(0.1),
        emiss=full(0.97), albedo=s2(inp("albedo")), sfcrunoff=zz, udrunoff=zz)
    table_dir = wrf_run_dir()
    static0 = NoahMPStatic(
        ivgtyp=jnp.asarray(ivg.reshape(shape), jnp.int32), isltyp=jnp.full(shape, 6, jnp.int32), xland=s2(xland),
        landmask=s2((xland < 1.5).astype(float)), lakemask=zz, lu_index=jnp.asarray(ivg.reshape(shape), jnp.int32),
        tbot=full(290.0), dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0]), zsoil=jnp.asarray([-0.1, -0.4, -1.0, -2.0]),
        lat=full(28.3), dx_m=1000.0, parameters=load_noahmp_parameters(table_dir), shdmax=full(0.4), shdfac=full(0.4))
    table = us.parse_urbparm(SAVE / "URBPARM.TBL.default")
    bundle = hook.UrbanBundle(table=table, isurban=13, natural=hook.mptable_natural(table_dir))
    static = hook.urban_noahmp_static(static0, bundle)
    carry = hook.initial_urban_carry(bundle, ivgtyp=static0.ivgtyp, tsk=s2(inp("tsk")), tslb=jnp.asarray(tslb),
                                     frc_urb2d=zz)

    class _Rad:
        soldn = s2(inp("swdown"))
        lwdn = s2(inp("glw"))
        cosz = s2(np.where(inp("swdown") > 0.0, 0.8, 0.0))

    class _Clock:
        julian = 59.0
        yearlen = 365.0

    return state, land, static, _Rad(), _Clock(), bundle, carry, ivg == 13


def test_adapter_routes_the_urban_blend_like_wrf(monkeypatch):
    """F1 (rv-urban): noahmp_surface_adapter(urban=...) vs the urban=None call on the same inputs.

    Urban cells must equal WRF's noahmp_urban (the oracle-validated hook) applied to the rural Noah-MP /
    sfclay outputs of the reference call: UST reaches MYNN, TSK/QSFC/ALBEDO feed the land carry, CHS (not CHS2)
    drives TS, and history T2/Q2/U10/V10 carry the surface-driver overwrite.  Every other cell is bit-identical.
    """
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.contracts.noahmp_state import NoahMPFluxes
    from gpuwrf.physics.noahmp.precision import real_tree
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing, noahmp_surface_adapter
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    if not (wrf_run_dir() / "MPTABLE.TBL").exists():
        pytest.skip("pristine WRF run/ tables not linked")
    monkeypatch.setenv("GPUWRF_LAND_Q2_CAP_POSTLSM", "0")
    state, land, static, rad, clock, bundle, carry, urb = _adapter_case()
    dt = 6.0
    ref = noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=dt, history=True,
                                 land_history=True)
    out = noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=dt, history=True,
                                 land_history=True, urban=(bundle, carry))
    assert len(out) == len(ref) + 1
    s_ref, l_ref, b_ref, f_ref = ref
    s_out, l_out, b_out, f_out, carry_out = out
    m = np.asarray(urb)
    a = lambda x: np.asarray(x, np.float64).reshape(-1)  # noqa: E731

    # --- non-urban cells: bit-identical in every routed output ---
    pairs = [(s_ref.t_skin, s_out.t_skin), (b_ref.ustar, b_out.ustar), (b_ref.theta_flux, b_out.theta_flux), (b_ref.qv_flux, b_out.qv_flux),
             (b_ref.fltv, b_out.fltv), (l_ref.t_skin, l_out.t_skin), (l_ref.qsfc, l_out.qsfc),
             (l_ref.albedo, l_out.albedo)] + [(f_ref[k], f_out[k]) for k in ("hfx", "lh", "t2", "q2", "u10", "v10",
                                                                             "grdflx")]
    for r_, o_ in pairs:
        np.testing.assert_array_equal(a(r_)[~m], a(o_)[~m])
    for r_, o_ in zip(jax.tree.leaves(l_ref), jax.tree.leaves(l_out)):
        np.testing.assert_array_equal(np.asarray(r_)[..., ~m.reshape(1, -1)], np.asarray(o_)[..., ~m.reshape(1, -1)])

    # --- urban cells: WRF noahmp_urban on the reference (rural) outputs ---
    diag = surface_layer_with_diagnostics(state, snowh=land.snowh, with_chs=True)
    assert not np.allclose(a(diag.chs)[m], a(diag.chs2)[m])     # CHS and CHS2 are different inputs
    forcing = real_tree(assemble_noahmp_forcing(state, static, rad, clock, dt))
    z = jnp.zeros_like(s_ref.t_skin)
    rural = NoahMPFluxes(hfx=f_ref["hfx"], lh=f_ref["lh"], qfx=s_ref.qfx if hasattr(s_ref, "qfx") else None,
                         grdflx=f_ref["grdflx"], tsk=l_ref.t_skin, qsfc=l_ref.qsfc, znt=z, emiss=z,
                         albedo=l_ref.albedo, chs=z)
    if rural.qfx is None:   # State namespace without a qfx slot: recover QFX from the kinematic handle
        rural = rural._replace(qfx=b_ref.qv_flux * diag.fluxes.rhosfc)
    nm_e, ust_e, carry_e, step_e = hook.apply_urban_slucm(
        bundle, carry, forcing=forcing, nm=rural, ust=b_ref.ustar, chs=diag.chs, chs2=diag.chs2, cqs2=diag.cqs2,
        qv_mixing_ratio=state.qv[..., 0], dt=dt)
    t2_e, q2_e, u10_e, v10_e = hook.urban_history_overwrite(f_ref["t2"], f_ref["q2"], f_ref["u10"], f_ref["v10"],
                                                            step_e, forcing.psfc)
    for name, got, exp in (("ustar->MYNN", b_out.ustar, ust_e), ("land t_skin", l_out.t_skin, nm_e.tsk),
                           ("land qsfc", l_out.qsfc, nm_e.qsfc), ("land albedo", l_out.albedo, nm_e.albedo),
                           ("state t_skin", s_out.t_skin, nm_e.tsk), ("hist hfx", f_out["hfx"], nm_e.hfx),
                           ("hist lh", f_out["lh"], nm_e.lh), ("hist grdflx", f_out["grdflx"], nm_e.grdflx),
                           ("hist t2", f_out["t2"], t2_e), ("hist q2", f_out["q2"], q2_e),
                           ("hist u10", f_out["u10"], u10_e), ("hist v10", f_out["v10"], v10_e),
                           ("carry tr", carry_out.state.tr, carry_e.state.tr)):
        np.testing.assert_allclose(a(got)[m], a(exp)[m], rtol=2e-6, atol=1e-9, err_msg=name)
    # the blend actually changed every routed urban value (a dropped hunk cannot hide behind equality)
    for name, r_, o_ in (("ustar", b_ref.ustar, b_out.ustar), ("t_skin", l_ref.t_skin, l_out.t_skin),
                         ("qsfc", l_ref.qsfc, l_out.qsfc), ("albedo", l_ref.albedo, l_out.albedo),
                         ("t2", f_ref["t2"], f_out["t2"]), ("u10", f_ref["u10"], f_out["u10"])):
        assert np.all(np.abs(a(o_)[m] - a(r_)[m]) > 0), name


def test_first_level_below_canopy_is_refused_like_wrf_fatal():
    """F3 (rv-urban): WRF module_sf_urban.F:826 FATAL 'ZDC + Z0C + 2m is larger than the 1st WRF level'."""
    table = us.parse_urbparm(SAVE / "URBPARM.TBL.default")
    bundle = hook.UrbanBundle(table=table, isurban=13, natural=14)
    limit = float(table.zdc[1]) + float(table.z0c[1]) + 2.0
    assert 8.0 < limit < 8.1                                    # UTYPE 2: ZDC+Z0C+2 = 8.05 m
    dz_ok = 2.0 * limit / (1.0 - hook.ZA_DRIFT_MARGIN) * 1.001   # just clears the margin
    ivg = np.array([[13, 10], [13, 5]])
    hook.check_first_level_above_canopy(bundle, ivgtyp=ivg, dz1=np.full((2, 2), dz_ok), domain="d01")
    hook.check_first_level_above_canopy(bundle, ivgtyp=ivg, dz1=np.array([[50.0, 2.0], [50.0, 2.0]]))  # non-urban thin
    for dz in (dz_ok * 0.995, 16.0, 12.0, 4.0):              # 12 m: U10 = 177 m/s in the unguarded kernel
        with pytest.raises(ValueError, match="module_sf_urban.F:826") as exc:
            hook.check_first_level_above_canopy(bundle, ivgtyp=ivg, dz1=np.array([[dz, 50.0], [50.0, 50.0]]),
                                                domain="d03")
        assert "d03" in str(exc.value) and "ZDC + Z0C + 2m is larger than the 1st WRF level" in str(exc.value)


def _fake_loader_inputs(ivg):
    from gpuwrf.config.paths import wrf_run_dir

    if not ((wrf_run_dir() / "URBPARM.TBL").exists() and (wrf_run_dir() / "MPTABLE.TBL").exists()):
        pytest.skip("pristine WRF run/ tables not linked")

    class _Static(SimpleNamespace):
        def replace(self, **u):
            d = dict(self.__dict__)
            d.update(u)
            return _Static(**d)

    ivg = np.asarray(ivg, np.int32)
    run = SimpleNamespace(wrfinput_variables=lambda domain: [],namelist={"physics":{}})
    land = SimpleNamespace(t_skin=jnp.full(ivg.shape, 290.0), tslb=jnp.full((4,) + ivg.shape, 288.0))
    static = _Static(ivgtyp=jnp.asarray(ivg), parameters=SimpleNamespace(isurban=13),
                     shdmax=jnp.full(ivg.shape, 0.4))
    return run, land, static


def test_nested_loader_applies_the_canopy_check_and_refuses_lcz():
    from gpuwrf.integration.nested_pipeline import _load_urban_slucm

    run, land, static = _fake_loader_inputs([[13, 10], [13, 5]])
    run.namelist = {"physics":{}}
    bundle, carry, static_u = _load_urban_slucm(run, "d01", land, static, dz1=np.full((2, 2), 51.0))
    assert np.asarray(static_u.ivgtyp).tolist() == [[14, 10], [14, 5]]
    assert np.asarray(carry.utype).tolist() == [[2, 0], [2, 0]]
    with pytest.raises(ValueError, match="module_sf_urban.F:826"):
        _load_urban_slucm(run, "d01", land, static, dz1=np.full((2, 2), 12.0))
    run, land, static = _fake_loader_inputs([[13, 53], [13, 5]])
    with pytest.raises(NotImplementedError, match="LCZ"):
        _load_urban_slucm(run, "d01", land, static, dz1=np.full((2, 2), 51.0))


def test_honour_check_covers_every_domain():
    from gpuwrf.io.namelist_binding import namelist_honour_report

    nl = {"physics": {"sf_urban_physics": [1, 1], "sf_surface_physics": [4, 2]}, "dynamics": {}, "noah_mp": {}}
    errs, _ = namelist_honour_report(nl, ("d01", "d02"))
    assert any("sf_urban_physics=1 with sf_surface_physics=2 (d02)" in e for e in errs), errs


@pytest.mark.parametrize("key", ["AHOPTION", "ALHOPTION"])
def test_operational_bundle_accepts_diurnal_anthropogenic_heat(tmp_path, key):
    src = (SAVE / "URBPARM.TBL.default").read_text()
    txt = "\n".join(f"{key}: 1" if line.startswith(f"{key}:") else line for line in src.splitlines())
    (tmp_path / "URBPARM.TBL").write_text(txt + "\n")
    from gpuwrf.config.paths import wrf_run_dir
    (tmp_path/"MPTABLE.TBL").symlink_to(wrf_run_dir()/"MPTABLE.TBL")
    bundle = hook.load_urban_bundle(tmp_path, isurban=13)
    assert getattr(bundle.table,key.lower())==1
