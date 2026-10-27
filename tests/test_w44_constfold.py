"""w44-constfold: WRF REAL constant folds on the REAL path (GPUWRF_WRF_REAL_CONSTFOLD).

WRF folds REAL constant expressions per operation in REAL; Python folds them in float64 and rounds once.
Pinned bits = the pristine gfortran 14.3 oracle (WRF 4.7.1 statements verbatim, PARAMETER chain from the
pristine build's own .mod files): proofs/w44_constfold/tools/wrf_residual_oracle.py ->
proofs/w44_constfold/wrf_residual_oracle.json.

Deletion sensitivity: (1) every REAL constant is pinned to the oracle bits, (2) the traced program of each
site carries WRF's literal with the flag ON and the old Python literal with the flag OFF or for float64
consumers, (3) an AST guard fails when any audited Python fold is used outside real_constant's legacy slot.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import jax
from jax._src import core as jcore
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts import real_constfold as rc

ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((ROOT / "proofs/w44_constfold/wrf_residual_oracle.json").read_text())


def _bits(value) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def _real_values():
    from gpuwrf.kernels import phys_kf_column as kf, rad_sw_band_sums as sw
    from gpuwrf.physics import mynn_edmf as me, mynn_pbl as mp, mynn_sgs_cloud as sc, surface_layer as sl
    from gpuwrf.physics.noahmp import energy as en, snow as sn
    return {
        "mynn_qsat_blend_chi_den": me._QSAT_CHI_DEN_REAL,
        "mynn_xl_blend_cpv_cice": sc._CPV_M_CICE_REAL,
        "mynn_xl_blend_chi_den": sc._T0C_M_TICE_REAL,
        "mynn_q1_rh_den": sc._ONE_M_RHCRIT_REAL,
        "mynn_level2_one_m_c2": mp._ONE_M_C2_REAL,
        "mynnsl_inv_27315": sl._INV_27315_REAL,
        "mynnsl_ep_3": sl._EP_3_REAL,
        "noahmp_vkc2": en._VKC2_REAL,
        "noahmp_rsurf_den": en._E_M1_REAL,
        "noahmp_snowh2o_ratio": sn._LIQ_RATIO_REAL,
        "kf_rhh_slope": kf._RHH_SLOPE_REAL,
        "rrtmg_sw_bpade": sw._BPADE_REAL,
    }


def test_oracle_is_finite_and_complete():
    names = set(_real_values()) | {"mynn_level2_f2"}
    assert names <= set(ORACLE)
    for name in names:
        value = np.asarray(ORACLE[name]["bits"], np.uint32).view(np.float32)
        assert np.isfinite(value) and np.isfinite(ORACLE[name]["value"]), name
        assert float(value) == ORACLE[name]["value"], name


@pytest.mark.parametrize("name", sorted(_real_values()))
def test_real_constant_is_wrf_real_fold(name):
    value = _real_values()[name]
    assert isinstance(value, np.float32) and np.isfinite(value)
    assert _bits(value) == ORACLE[name]["bits"], (name, float(value), ORACLE[name]["value"])


def test_mynn_level2_f2_composite_is_wrf_bits():
    """f2 = b1*(g1+g2) - 3.0*a1*(one-c2) is an all-PARAMETER REAL expression (module_bl_mynnedmf.F90:1557)."""
    from gpuwrf.physics import mynn_pbl as mp
    from gpuwrf.physics.mynn_constants import B1, G1, _WRF_REAL_VALUES as W
    f2 = B1 * (G1 + W["G2"]) - 3.0 * W["A1"] * mp._ONE_M_C2_REAL
    assert isinstance(f2, np.float32) and _bits(f2) == ORACLE["mynn_level2_f2"]["bits"]
    old = B1 * (G1 + W["G2"]) - 3.0 * W["A1"] * (1.0 - mp.C2)
    assert _bits(old) != ORACLE["mynn_level2_f2"]["bits"]  # the Python (1.0-C2) fold is NOT WRF's f2


def _f32_literals(fn, *args):
    jax.clear_caches()
    jaxpr = jax.make_jaxpr(fn)(*args)
    found = set()

    def walk(jx):
        for eqn in jx.eqns:
            for v in eqn.invars:
                if isinstance(v, jcore.Literal) and np.asarray(v.val).dtype == np.float32:
                    found.add(_bits(np.asarray(v.val)))
            for p in eqn.params.values():
                for sub in (p if isinstance(p, (tuple, list)) else (p,)):
                    if hasattr(sub, "jaxpr") and hasattr(sub, "consts"):
                        walk(sub.jaxpr)
                    elif hasattr(sub, "eqns"):
                        walk(sub)
    walk(jaxpr.jaxpr)
    for c in jaxpr.consts:
        arr = np.asarray(c)
        if arr.dtype == np.float32 and arr.size <= 16:
            found |= {_bits(x) for x in arr.ravel()}
    return found


def _site_cases():
    from gpuwrf.kernels import rad_sw_band_sums as sw
    from gpuwrf.physics import mynn_edmf as me, mynn_sgs_cloud as sc
    from gpuwrf.physics.noahmp import energy as en

    def qsat(t):
        return me._qsat_blend(t, jnp.full_like(t, 80000.0))

    def sfcdif(z):
        one = jnp.ones_like(z)
        return en.sfcdif1(0, 280.0 * one, 1.2 * one, 10.0 * one, 0.005 * one, 10.0 * one, 0.0 * one, 0.1 * one,
                          0.01 * one, 3.0 * one, 0.0 * one, 0.0 * one, 0.0 * one, 0.0 * one, 0.0 * one, 0.0 * one,
                          0.3 * one)[0]

    def cloudpdf2(t):
        nz = t.shape[-1]
        zero = jnp.zeros_like(t)
        return sc.mym_condensation_cloudpdf2(theta=t, p=jnp.full_like(t, 50000.0), exner=jnp.ones_like(t),
                                             dz=jnp.full_like(t, 200.0), qw=zero + 1e-3, qc=zero + 2e-6,
                                             qi=zero + 1e-8, qs=zero, qsq=zero + 1e-8,
                                             pblh=jnp.full(t.shape[:-1], 100.0, t.dtype))[2] * nz

    return {
        "mynn_qsat_blend_chi_den": (qsat, "(T0C - 6.0) - TICE", me),
        "mynn_xl_blend_cpv_cice": (sc.xl_blend, "CPV - CICE", sc),
        "mynn_xl_blend_chi_den": (sc.xl_blend, "T0C - TICE", sc),
        "mynn_q1_rh_den": (cloudpdf2, "1.0 - RHCRIT", sc),
        "noahmp_vkc2": (sfcdif, "VKC * VKC", en),
        "rrtmg_sw_bpade": (sw._lookup, "1.0 / 0.278", None),
    }


@pytest.mark.parametrize("name", sorted(_site_cases()))
def test_traced_site_selects_wrf_real_only_for_real_consumers(monkeypatch, name):
    fn, legacy_src, module = _site_cases()[name]
    legacy_value = eval(legacy_src, vars(module)) if module is not None else 1.0 / 0.278
    real_bits, legacy_bits = ORACLE[name]["bits"], _bits(legacy_value)
    assert real_bits != legacy_bits  # the audited site is a genuine REAL-fold difference
    t32 = jnp.linspace(235.0, 275.0, 12, dtype=jnp.float32).reshape(2, 6)
    monkeypatch.setattr(rc, "ENABLED", True)
    on = _f32_literals(fn, t32)
    assert real_bits in on and legacy_bits not in on, name
    monkeypatch.setattr(rc, "ENABLED", False)
    off = _f32_literals(fn, t32)
    assert legacy_bits in off and real_bits not in off, name
    if module is not None:  # float64 (legacy/DOUBLE) consumers keep the Python value under the flag
        monkeypatch.setattr(rc, "ENABLED", True)
        with jax.enable_x64(True):
            assert rc.real_constant(np.float32(0), legacy_value, jnp.result_type(t32.astype(jnp.float64))) == legacy_value


# Every audited Python fold must only appear as real_constant's legacy (2nd) argument.
AUDITED = {
    "src/gpuwrf/physics/mynn_edmf.py": ["(T0C - 6.0) - TICE"],
    "src/gpuwrf/physics/mynn_sgs_cloud.py": ["CPV - CICE", "T0C - TICE", "1.0 - RHCRIT"],
    "src/gpuwrf/physics/mynn_pbl.py": ["1.0 - C2"],
    "src/gpuwrf/physics/surface_layer.py": ["1.0 - EP2", "1.0 / 273.15"],
    "src/gpuwrf/physics/noahmp/energy.py": ["VKC * VKC", "2.71828 - 1.0"],
    "src/gpuwrf/physics/noahmp/snow.py": ["MAX_LIQ_MASS_FRACTION / (1.0 - MAX_LIQ_MASS_FRACTION)"],
    "src/gpuwrf/kernels/phys_kf_column.py": ["0.2 / 1000.0"],
    "src/gpuwrf/kernels/rad_sw_band_sums.py": ["1.0 / 0.278"],
}


@pytest.mark.parametrize("path", sorted(AUDITED))
def test_audited_folds_only_as_legacy_argument(path):
    tree = ast.parse((ROOT / path).read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for src in AUDITED[path]:
        target = ast.dump(ast.parse(src, mode="eval").body)
        hits = [n for n in ast.walk(tree) if isinstance(n, ast.expr) and ast.dump(n) == target]
        assert hits, (path, src)
        guarded = 0
        for node in hits:
            par = parents.get(node)
            ok = (isinstance(par, ast.Call) and getattr(par.func, "id", None) == "real_constant"
                  and len(par.args) >= 2 and par.args[1] is node)
            assert ok, f"{path}:{node.lineno} Python fold {src!r} outside real_constant(..., legacy)"
            guarded += 1
        assert guarded == len(hits)


@pytest.mark.parametrize("dts,name", [(4.5, "acoustic_dampmag_dts4p5"), (18.0, "acoustic_dampmag_dts18"),
                                      (13.5, "acoustic_dampmag_dts13p5")])
def test_acoustic_dampmag_is_wrf_real_product(monkeypatch, dts, name):
    """advance_w dampmag = dts*config_flags%dampcoef is a REAL run-time product (module_small_step_em.F:1284)."""
    from gpuwrf.kernels import dyn_acoustic_fp32 as ac
    monkeypatch.setattr(rc, "ENABLED", True)
    on = ac._dampmag(dts, 0.2)
    assert isinstance(on, np.float32) and _bits(on) == ORACLE[name]["bits"]
    monkeypatch.setattr(rc, "ENABLED", False)
    off = ac._dampmag(dts, 0.2)
    assert off == dts * 0.2 and isinstance(off, float)
    if name != "acoustic_dampmag_dts13p5":  # the Swiss/PROD-d01 stage values are genuine 1-ulp differences
        assert _bits(off) != ORACLE[name]["bits"]
    # a traced (runtime-controlled, w44-params) dampcoef is never refolded on the host: one traced REAL product
    monkeypatch.setattr(rc, "ENABLED", True)
    jx = jax.make_jaxpr(lambda c: ac._dampmag(dts, c))(jnp.float32(0.2))
    assert [e.primitive.name for e in jx.jaxpr.eqns] == ["mul"]


@pytest.mark.parametrize("dt,name", [(54.0, "noahmp_class_alb_arg_dt54"), (18.0, "noahmp_class_alb_arg_dt18")])
def test_noahmp_class_albedo_argument_is_wrf_real(monkeypatch, dt, name):
    """EXP(-0.01*DT/3600.0) argument (module_sf_noahmplsm.F:3261) is a REAL run-time expression of DT."""
    from gpuwrf.physics.noahmp import energy_radiation as er, precision as pr
    real_bits = ORACLE[name]["bits"]
    monkeypatch.setattr(rc, "ENABLED", True)
    monkeypatch.setattr(er, "native_real_enabled", lambda: True)  # native REAL Noah-MP (release default)
    monkeypatch.setattr(pr, "native_real_enabled", lambda: True)  # real_scalar() consults its own module flag
    albold = jnp.full((3,), 0.7, jnp.float32)
    lits = _f32_literals(lambda a: er._snowalb_class_value(jnp.zeros_like(a), dt, a, _ClassParams()), albold)
    assert real_bits in lits
    legacy = _bits(-0.01 * dt / 3600.0)
    if legacy != real_bits:
        assert legacy not in lits
        monkeypatch.setattr(rc, "ENABLED", False)
        assert legacy in _f32_literals(lambda a: er._snowalb_class_value(jnp.zeros_like(a), dt, a, _ClassParams()), albold)


class _ClassParams:
    swemx = 1.0


@pytest.mark.parametrize("dt,name", [(54.0, "noahmp_snow_ret_dt54"), (6.0, "noahmp_snow_ret_dt6"), (18.0, "noahmp_snow_ret_dt18")])
def test_noahmp_snow_release_product_is_wrf_real(monkeypatch, dt, name):
    """SNOW_RET_FAC*DT (module_sf_noahmplsm.F:7211) is a REAL run-time product; traced dt is left alone."""
    from gpuwrf.physics.noahmp import snow as sn
    monkeypatch.setattr(rc, "ENABLED", True)
    on = rc.real_scalar_product(sn.SNOW_RET_FAC, dt, jnp.float32)
    assert isinstance(on, np.float32) and _bits(on) == ORACLE[name]["bits"]
    assert rc.real_scalar_product(sn.SNOW_RET_FAC, dt, jnp.float64) == sn.SNOW_RET_FAC * dt
    traced = jax.make_jaxpr(lambda d: rc.real_scalar_product(sn.SNOW_RET_FAC, d, jnp.float32))(jnp.float32(dt))
    assert any(e.primitive.name == "mul" for e in traced.jaxpr.eqns)
    monkeypatch.setattr(rc, "ENABLED", False)
    assert rc.real_scalar_product(sn.SNOW_RET_FAC, dt, jnp.float32) == sn.SNOW_RET_FAC * dt


def test_snow_release_sites_use_the_real_product():
    tree = ast.parse((ROOT / "src/gpuwrf/physics/noahmp/snow.py").read_text())
    raw = ast.dump(ast.parse("SNOW_RET_FAC * dt", mode="eval").body)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.expr) and ast.dump(n) == raw]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "real_scalar_product"]
    assert len(calls) == 2 and all(ast.unparse(c.args[0]) == "SNOW_RET_FAC" and ast.unparse(c.args[1]) == "dt" for c in calls)


def test_acoustic_dampmag_site_uses_the_real_product():
    tree = ast.parse((ROOT / "src/gpuwrf/kernels/dyn_acoustic_fp32.py").read_text())
    raw = ast.dump(ast.parse("cfg.dt * cfg.dampcoef", mode="eval").body)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.expr) and ast.dump(n) == raw]
    sites = [k for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "dampmag"]
    assert sites and all(isinstance(k.value, ast.Call) and getattr(k.value.func, "id", None) == "_dampmag" for k in sites)


def test_flag_is_in_cheap_key_registry():
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    assert ("gpuwrf.contracts.real_constfold", "ENABLED") in IMPORT_TIME_ENV_CONSTANTS


# ---- F3 (cr-w44-constfold): wiring pins - every REAL-fold call with its REAL value and consumer-dtype expression ----
WIRING = {
    "src/gpuwrf/physics/mynn_edmf.py": [("real_constant", "_QSAT_CHI_DEN_REAL", "jnp.result_type(t)")],
    "src/gpuwrf/physics/mynn_sgs_cloud.py": [("real_constant", "_CPV_M_CICE_REAL", "dtype"),
                                             ("real_constant", "_T0C_M_TICE_REAL", "dtype"),
                                             ("real_constant", "_ONE_M_RHCRIT_REAL", "jnp.result_type(rh)"),
                                             ("real_constant", "_ONE_M_RHCRIT_REAL", "jnp.result_type(rh)")],
    "src/gpuwrf/physics/mynn_pbl.py": [("real_constant", "_ONE_M_C2_REAL", "state.theta.dtype")],
    "src/gpuwrf/physics/surface_layer.py": [("real_constant", "_EP_3_REAL", "F"),
                                            ("real_constant", "_INV_27315_REAL", "F")],
    "src/gpuwrf/physics/noahmp/energy.py": [("real_constant", "_VKC2_REAL", "jnp.result_type(cmfm)"),
                                            ("real_constant", "_E_M1_REAL", "jnp.result_type(smcmax1)")],
    "src/gpuwrf/physics/noahmp/snow.py": [("real_constant", "_LIQ_RATIO_REAL", "jnp.result_type(like)"),
                                          ("real_scalar_product", "SNOW_RET_FAC", "jnp.result_type(qout_k)"),
                                          ("real_scalar_product", "SNOW_RET_FAC", "jnp.result_type(qout_k)"),
                                          ("_liq_ratio", "snice[k]", "<default>"), ("_liq_ratio", "snice[k]", "<default>"),
                                          ("_liq_ratio", "I[k]", "<default>"), ("_liq_ratio", "I[k]", "<default>")],
    "src/gpuwrf/physics/noahmp/energy_radiation.py": [
        ("real_constant", "R4(-0.01) * R4(dt) / R4(3600.0)", "jnp.float32 if native_real_enabled() else jnp.float64"),
        ("_class_alb_arg", "dt", "<default>")],
    "src/gpuwrf/kernels/phys_kf_column.py": [("real_constant", "_RHH_SLOPE_REAL", "<default>")],
    "src/gpuwrf/kernels/rad_sw_band_sums.py": [("real_constant", "_BPADE_REAL", "<default>")],
    "src/gpuwrf/kernels/dyn_acoustic_fp32.py": [("real_scalar_product", "dts", "<default>"), ("_dampmag", "cfg.dt", "<default>")],
}


@pytest.mark.parametrize("path", sorted(WIRING))
def test_real_fold_wiring_is_pinned(path):
    """A dtype argument flipped to float64 or a different value in the REAL slot silently restores the old constant."""
    tree = ast.parse((ROOT / path).read_text())
    names = {name for name, _real, _dtype in WIRING[path]}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in names:
            dtype = ast.unparse(node.args[2]) if len(node.args) > 2 else next(
                (ast.unparse(k.value) for k in node.keywords if k.arg == "dtype"), "<default>")
            found.append((node.func.id, ast.unparse(node.args[0]), dtype))
    assert sorted(found) == sorted(WIRING[path]), path


def test_mynn_level2_traced_constants(monkeypatch):
    """_mym_level2 on real MYNN columns: (one-c2) in f1 and the all-REAL f2 carry WRF's bits (flag ON, REAL state)."""
    import sys
    sys.path.insert(0, str(ROOT / "tests/v025/b_phys"))
    from test_mynn_qni import DATA, state_from_row
    from gpuwrf.physics import mynn_constants as MC, mynn_pbl as mp
    monkeypatch.setattr(MC, "_MYNN_REAL_CONSTANTS", True)  # release default (BD94 typed a1/c1/a2/g2)
    state = state_from_row(DATA["rows"][1])
    run = lambda s: mp._mym_level2(s)  # noqa: E731
    real, f2 = ORACLE["mynn_level2_one_m_c2"]["bits"], ORACLE["mynn_level2_f2"]["bits"]
    legacy = _bits(1.0 - mp.C2)
    monkeypatch.setattr(rc, "ENABLED", True)
    on = _f32_literals(run, state)
    assert real in on and f2 in on and legacy not in on
    monkeypatch.setattr(rc, "ENABLED", False)
    off = _f32_literals(run, state)
    assert legacy in off and real not in off and f2 not in off


def test_mynn_surface_layer_traced_constants(monkeypatch):
    """MYNN-SL _surface_layer_impl (isftcflx=0, the release path): ep_3 and 1./273.15 carry WRF's bits."""
    from types import SimpleNamespace
    from gpuwrf.physics import surface_layer as sl
    n = 3
    col = lambda v: jnp.full((n, 1), v, jnp.float32)  # noqa: E731
    lev = lambda v: jnp.full((n, 1, 1), v, jnp.float32)  # noqa: E731

    def run(tsk):
        view = SimpleNamespace(rho=col(1.1), mavail=col(0.5), pblh=col(500.0), xland=jnp.asarray([[1.0], [2.0], [1.0]], jnp.float32),
                               psfc=col(95000.0), mol=col(0.0), qsfc=col(0.008), hfx=col(10.0), qfx=col(1e-5),
                               t_skin=tsk, roughness_m=col(0.1), ustar=col(0.3), u=lev(3.0), v=lev(1.0),
                               theta=lev(285.0), t_air=lev(281.0), qv=lev(0.006), p=lev(94000.0), dz=lev(50.0),
                               lakemask=col(0.0), dx_m=9000.0)
        return sl._surface_layer_impl(view, False, jnp.float32)
    tsk = jnp.asarray([[268.0], [284.0], [290.0]], jnp.float32)
    pins = [("mynnsl_ep_3", 1.0 - sl.EP2), ("mynnsl_inv_27315", 1.0 / 273.15)]
    monkeypatch.setattr(rc, "ENABLED", True)
    on = _f32_literals(run, tsk)
    monkeypatch.setattr(rc, "ENABLED", False)
    off = _f32_literals(run, tsk)
    for label, legacy in pins:
        assert ORACLE[label]["bits"] in on and _bits(legacy) not in on, label
        assert _bits(legacy) in off and ORACLE[label]["bits"] not in off, label


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dampmag_runtime_parameter_real_fold(monkeypatch, dtype):
    """Joint rule with w44-params: dampcoef as a TRACED runtime operand is never converted on the host; with the flag
    ON it is cast to REAL and multiplied once in REAL (WRF dts*dampcoef), with the flag OFF the product is unchanged."""
    from gpuwrf.kernels import dyn_acoustic_fp32 as ac
    with jax.enable_x64(dtype == jnp.float64):
        coef = jnp.asarray(0.2, dtype)
        monkeypatch.setattr(rc, "ENABLED", True)
        on = jax.jit(lambda c: ac._dampmag(4.5, c))(coef)
        assert on.dtype == jnp.float32 and _bits(on) == ORACLE["acoustic_dampmag_dts4p5"]["bits"]
        monkeypatch.setattr(rc, "ENABLED", False)
        off = jax.jit(lambda c: ac._dampmag(4.5, c))(coef)
        assert off.dtype == dtype
        assert np.asarray(off) == np.asarray(4.5 * coef)


@pytest.mark.parametrize("flag", [True, False])
def test_class_albedo_keeps_main_trace_order_for_traced_dt(monkeypatch, flag):
    """GPU Noah-MP radiation kernel: DT is traced; (ALBOLD-0.55) must be traced before -0.01*DT/3600 (critic F2/CM8),
    i.e. the first equations equal main's ALB = 0.55 + (ALBOLD-0.55)*EXP(-0.01*DT/3600.0)."""
    from gpuwrf.physics.noahmp import energy_radiation as er, precision as pr
    monkeypatch.setattr(rc, "ENABLED", flag)
    monkeypatch.setattr(er, "native_real_enabled", lambda: True)
    monkeypatch.setattr(pr, "native_real_enabled", lambda: True)
    albold = jnp.full((3,), 0.7, jnp.float32)
    dt = jnp.float32(54.0)
    prims = lambda f: [e.primitive.name for e in jax.make_jaxpr(f)(albold, dt).jaxpr.eqns]  # noqa: E731
    qsnow = jnp.zeros((3,), jnp.float32)  # concrete closure: no leading broadcast equation
    full = prims(lambda a, d: er._snowalb_class_value(qsnow, d, a, _ClassParams()))
    main = prims(lambda a, d: 0.55 + (a - 0.55) * jnp.exp(pr.real_scalar(-0.01 * d / 3600.0)))
    assert full[:len(main)] == main, (full[:8], main)
