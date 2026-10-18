"""WRF 4.7.1 radiation cloud fraction icloud=3 (cal_cldfra3) and icloud=2 (cal_cldfra2).

Savepoint = verbatim pristine outputs + independently frozen FMA/input-ULP envelope
(proofs/w41_icloud/oracle_gate.py) on regime-censused real columns
(proofs/w41_icloud/fixtures/icloud3_columns_v1.npz).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_v022_camuw_pbl import _grid, _namelist, _state  # noqa: E402

from gpuwrf.coupling import physics_couplers as pc  # noqa: E402
from gpuwrf.io.namelist_binding import icloud_refusals, namelist_honour_report  # noqa: E402
from gpuwrf.physics import cloud_fraction as cf_mod  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

PROOFS = Path(__file__).resolve().parents[2] / "proofs/w41_icloud"
FIXTURE = PROOFS / "fixtures/icloud3_columns_v1.npz"
SAVEPOINT = PROOFS / "fixtures/icloud3_savepoint_v1.npz"
NAMES = ("qv", "qc", "qi", "qs", "dz", "p", "t")
OUTS = ("cldfra", "qc", "qi")


@pytest.fixture(scope="module")
def data():
    rows = np.load(FIXTURE)
    saved = np.load(SAVEPOINT)
    assert str(saved["fixture_sha256"]) == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    return {k: rows[k] for k in (*NAMES, "xland", "gridkm")}, {k: saved[k] for k in saved.files}


def _host_log10(x):
    # glibc 2.43 log10f (the oracle's libm) is correctly rounded: f64 log10 rounded once.
    return jnp.asarray(np.log10(np.asarray(x, np.float64)).astype(np.float32))


def _groups(rows):
    for g in np.unique(rows["gridkm"]):
        sel = rows["gridkm"] == g
        yield float(g), sel, [jnp.asarray(rows[n][sel]) for n in (*NAMES, "xland")]


def test_savepoint_provenance_is_verbatim_pristine(data):
    _, saved = data
    info = json.loads(str(saved["provenance"]))
    assert info["base"]["flags"][:2] == ["-O2", "-ffp-contract=off"]
    assert "-mfma" in info["fma"]["flags"]
    assert set(info["base"]["sha256"]) >= {"cal_cldfra3..adjust_cloudFinal", "RSLF", "RSIF"}
    assert info["cldfra2"]["sha256"]
    assert saved["cldfra"].shape == (640, 44)
    assert np.isfinite(saved["cldfra"]).all() and np.isfinite(saved["qi"]).all()


def test_literal_port_is_byte_identical_to_pristine_cal_cldfra3(data, monkeypatch):
    rows, saved = data
    monkeypatch.setattr(cf_mod, "_LOG10", _host_log10)
    for g, sel, args in _groups(rows):
        with jax.disable_jit():
            got = cf_mod.wrf_cal_cldfra3(*args, gridkm=g)
        for name, value in zip(OUTS, got):
            value = np.asarray(value)
            assert value.dtype == np.float32
            np.testing.assert_array_equal(value.view(np.int32), saved[name][sel].view(np.int32), err_msg=name)


def test_compiled_port_within_wrf_fma_and_input_ulp_envelope(data, monkeypatch):
    rows, saved = data
    monkeypatch.setattr(cf_mod, "_LOG10", lambda x: jax.pure_callback(
        lambda v: np.asarray(_host_log10(v)), jax.ShapeDtypeStruct(x.shape, jnp.float32), x,
        vmap_method="broadcast_all"))
    for g, sel, args in _groups(rows):
        got = jax.jit(lambda *a, g=g: cf_mod.wrf_cal_cldfra3(*a, gridkm=g))(*args)
        for name, value in zip(OUTS, got):
            delta = np.abs(np.asarray(value, np.float64) - saved[name][sel])
            assert np.all(delta <= saved[f"{name}_bound"][sel]), (name, float(delta.max()))


def test_product_xla_log10_deviation_is_confined_to_small_condensate_branch(data):
    rows, saved = data
    for g, sel, args in _groups(rows):
        cf, qc, qi, diag = jax.jit(lambda *a, g=g: cf_mod.wrf_cal_cldfra3(*a, gridkm=g, return_diagnostics=True))(*args)
        ref = saved["cldfra"][sel]
        outside = np.abs(np.asarray(cf, np.float64) - ref) > saved["cldfra_bound"][sel]
        assert not np.any(outside & ~np.asarray(diag["small"]))
        ulps = np.abs(np.asarray(cf).view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64))
        assert ulps[outside].max(initial=0) <= 16
        for name, value in (("qc", qc), ("qi", qi)):
            delta = np.abs(np.asarray(value, np.float64) - saved[name][sel])
            assert np.all(delta <= saved[f"{name}_bound"][sel] + 1e-5 * np.abs(saved[name][sel])), name


def test_every_cal_cldfra3_branch_is_active_on_the_fixture(data):
    rows, _ = data
    seen = {}
    for g, _sel, args in _groups(rows):
        *_, diag = jax.jit(lambda *a, g=g: cf_mod.wrf_cal_cldfra3(*a, gridkm=g, return_diagnostics=True))(*args)
        for key, value in diag.items():
            value = np.asarray(value)
            if value.dtype == bool:
                seen[key] = seen.get(key, 0) + int(value.any(axis=-1).sum() if value.ndim == 2 else value.sum())
    for key in ("full", "small", "rh", "ice_multi", "ice_single", "h2o_multi", "h2o_single",
                "lwp_scaled", "iwp_scaled", "top_cleared"):
        assert seen[key] > 0, key


@pytest.mark.parametrize("has_qi,column", [(True, "cldfra2_qc_qi"), (False, "cldfra2_qc_only")])
def test_cal_cldfra2_byte_identical_to_pristine(data, has_qi, column):
    rows, saved = data
    got = np.asarray(jax.jit(lambda a, b: cf_mod.wrf_cal_cldfra2(a, b, has_qi=has_qi))(rows["qc"], rows["qi"]))
    np.testing.assert_array_equal(got, saved[column])
    assert 0 < saved[column].mean() < 1


def test_icloud_selects_radiation_cloud_policy_and_default_is_untouched():
    grid = _grid()
    assert om._rrtmg_pbl_cloud_kwargs(_namelist(grid, bl_pbl_physics=5)) == {}
    for bl in (5, 9):
        for icloud in (2, 3):
            kwargs = om._rrtmg_pbl_cloud_kwargs(_namelist(grid, bl_pbl_physics=bl, icloud=icloud))
            assert kwargs == {"use_mynn_sgs_cloud": False, "cloud_mp_physics": 8, "icloud": icloud}
    with pytest.raises(ValueError, match="RRTMG"):
        om._rrtmg_pbl_cloud_kwargs(_namelist(grid, icloud=3, ra_lw_physics=1, ra_sw_physics=1))


def test_icloud_option_keys_the_static_program():
    grid = _grid()
    default, three = _namelist(grid), _namelist(grid, icloud=3)
    leaves, treedef = jax.tree_util.tree_flatten(three)
    assert treedef != jax.tree_util.tree_structure(default)
    assert jax.tree_util.tree_unflatten(treedef, leaves).icloud == 3


@pytest.mark.parametrize("edits,ok", [
    ({"icloud": 3}, True), ({"icloud": 2}, True), ({"icloud": 1}, True),
    ({"icloud": 0}, False), ({"icloud": 3, "mp_physics": 3}, False), ({"icloud": 3, "mp_physics": 1}, False),
    ({"icloud": 3, "ra_lw_physics": 1, "ra_sw_physics": 1}, False), ({"icloud": 3, "cu_physics": 10}, False),
    ({"icloud": 3, "mp_physics": 11}, False), ({"icloud": 2, "mp_physics": 3}, True),
    ({"icloud": 3, "cu_physics": 11}, True),
])
def test_icloud_honour_report(edits, ok):
    nml = {"physics": {"mp_physics": [8], "ra_lw_physics": [4], "ra_sw_physics": [4], "bl_pbl_physics": [5],
                       "cu_physics": [0]}}
    for key, value in edits.items():
        nml["physics"][key] = value if key == "icloud" else [value]
    errors = [e for e in namelist_honour_report(nml, ("d01",))[0] if "icloud" in e]
    assert (not errors) is ok, errors
    direct = icloud_refusals(edits["icloud"], mp_physics=nml["physics"]["mp_physics"][0],
                             cu_physics=nml["physics"]["cu_physics"][0], ra_lw_physics=nml["physics"]["ra_lw_physics"][0],
                             ra_sw_physics=nml["physics"]["ra_sw_physics"][0])
    assert (not direct) is ok


def _real_column_state(data):
    """Nine real Swiss/Canary columns as a 3x3x44 State on the caller-test grid (dx = dy = 3 km)."""
    rows, _ = data
    same = np.flatnonzero(np.isclose(rows["gridkm"], cf_mod.wrf_gridkm(3000., 3000.)))
    pick = same[np.linspace(0, same.size - 1, 9).astype(int)]
    grid = _grid(nz=44)
    col = lambda name: jnp.asarray(np.moveaxis(rows[name][pick].reshape(3, 3, 44), -1, 0), jnp.float64)  # noqa: E731
    p, t, qv = col("p"), col("t"), col("qv")
    rcp = 287.0 / 1004.5
    theta_m = t * (100000.0 / p) ** rcp * (1.0 + 461.6 / 287.0 * qv)
    ph = jnp.concatenate([jnp.zeros((1, 3, 3)), jnp.cumsum(col("dz") * 9.81, axis=0)], axis=0)
    state = _state(grid).replace(theta=theta_m, p=p, p_total=p, ph=ph, ph_total=ph, qv=qv, qc=col("qc"),
                                 qi=col("qi"), qs=col("qs"), xland=jnp.asarray(rows["xland"][pick].reshape(3, 3)))
    return grid, state


def _literal_gridkm(dx, dy):
    km, dx, dy = np.float32(0.001), np.float32(dx), np.float32(dy)
    return np.sqrt(np.float32(dx * km * dx * km + dy * km * dy * km))


def test_real_call_path_feeds_rrtmg_the_wrf_icloud3_caller_operands(data):
    grid, state = _real_column_state(data)
    kwargs = dict(time_utc="2023-01-15T04:00:00Z", lead_seconds=3600., use_mp_re=0)
    # Independent caller wiring (first_rk_step_part1 -> radiation_driver :1470-1500):
    # T = t_phy, P = p_hyd, DZ8W from (ph+phb)/9.81, XLAND, gridkm = SQRT(dx^2+dy^2) km.
    cols = lambda a: jnp.moveaxis(a, 0, -1)  # noqa: E731
    # Lower-level vapour near saturation of the caller operands -> RH-branch partial clouds,
    # the only branch that reads gridkm (scale-aware RH_00) and dz8w (delz).
    p_hyd0, _w, _s = pc._wrf_hydrostatic_pressure_profiles_from_state(state, grid.metrics)
    sw0, *_ = pc._rrtmg_column_inputs(state, grid, **kwargs, use_mynn_sgs_cloud=False, cloud_mp_physics=8, icloud=3)
    qvs = jnp.moveaxis(cf_mod._rslf(cols(p_hyd0).astype(jnp.float32), sw0.T.astype(jnp.float32)), -1, 0)
    factor = jnp.where(jnp.arange(44)[:, None, None] % 2 == 0, 0.97, 0.995)
    state = state.replace(qv=jnp.where(jnp.arange(44)[:, None, None] < 14, factor * qvs, state.qv).astype(state.qv.dtype),
                          qc=jnp.zeros_like(state.qc), qi=jnp.zeros_like(state.qi))
    sw, lw, *_ = pc._rrtmg_column_inputs(state, grid, **kwargs, use_mynn_sgs_cloud=False, cloud_mp_physics=8, icloud=3)
    # p_hyd is the MOIST hydrostatic pressure: evaluate it on the state radiation sees.
    p_hyd, _w, _s = pc._wrf_hydrostatic_pressure_profiles_from_state(state, grid.metrics)
    z_w = jnp.asarray(state.ph, jnp.float32) / jnp.float32(9.81)
    exp = cf_mod.wrf_cal_cldfra3(cols(state.qv), cols(state.qc), cols(state.qi), cols(state.qs),
                                 cols(z_w[1:] - z_w[:-1]), cols(p_hyd), sw.T, state.xland,
                                 _literal_gridkm(3000., 3000.), return_diagnostics=True)
    for column_state in (sw, lw):
        np.testing.assert_array_equal(np.asarray(column_state.cloud_fraction), np.asarray(exp[0]))
        np.testing.assert_array_equal(np.asarray(column_state.qc, np.float32), np.asarray(exp[1]))
        np.testing.assert_array_equal(np.asarray(column_state.qi, np.float32), np.asarray(exp[2]))
    cf = np.asarray(exp[0])
    changed = [np.sum(np.asarray(exp[i]) != np.asarray(cols(q), np.float32)) for i, q in ((1, state.qc), (2, state.qi))]
    assert np.any((cf > 0) & (cf < 1)) and sum(changed) > 0, changed  # fake radiation condensate is active
    rh_partial = np.asarray(exp[3]["rh"]) & (cf > 0) & (cf < 1)
    assert rh_partial.sum() >= 9, rh_partial.sum()  # gridkm/dz8w-sensitive RH branch is exercised
    # The MYNN BL cloud merge is part of WRF's icloud=1 block only: BL clouds must not leak in.
    cloudy = state.replace(cldfra_bl=jnp.full_like(state.qv, .5), qc_bl=jnp.full_like(state.qv, 1e-4),
                           qi_bl=jnp.full_like(state.qv, 1e-5))
    sw_bl, *_ = pc._rrtmg_column_inputs(cloudy, grid, **kwargs, use_mynn_sgs_cloud=False, cloud_mp_physics=8, icloud=3)
    np.testing.assert_array_equal(np.asarray(sw_bl.cloud_fraction), cf)
    np.testing.assert_array_equal(np.asarray(sw_bl.qc), np.asarray(sw.qc))
    # History CLDFRA is the radiation input fraction (held radiation diagnostics).
    class _Flux:  # solver stand-in: this seam test is about the held cloud field, not RRTMG fluxes
        def __getattr__(self, name):
            return jnp.zeros((3, 3))

    diag = pc.rrtmg_radiation_diagnostics(state, grid, **kwargs, use_mynn_sgs_cloud=False, cloud_mp_physics=8,
                                          icloud=3, _kernel_call=lambda fn, *a, **k: _Flux())
    np.testing.assert_array_equal(np.asarray(diag.cloud_fraction), np.moveaxis(cf, -1, 0))


def test_operational_driver_forwards_icloud(monkeypatch):
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=5, use_noahmp=False, mp_physics=8, icloud=3)
    carry = om._initial_carry_for_run(_state(grid), nml)
    calls = []

    def radiation(*args, **kwargs):
        calls.append(kwargs)
        return carry.rthraten, carry.radiation_diagnostics

    monkeypatch.setattr(om, "rrtmg_theta_tendency", radiation)
    om._refresh_rrtmg_driver(carry, nml, 3600., True, None)
    assert calls[-1]["icloud"] == 3 and calls[-1]["use_mynn_sgs_cloud"] is False
    om._refresh_rrtmg_driver(carry, dataclasses.replace(nml, icloud=1), 3600., True, None)
    assert "icloud" not in calls[-1] and "use_mynn_sgs_cloud" not in calls[-1]


def test_pert_cld3_multi_perturbation_of_icloud3_inputs_is_refused():
    # radiation_driver.F:1475-1478 perturbs the cal_cldfra3 T/QV inputs; the &stoch gate refuses it.
    nml = {"physics": {"mp_physics": [8], "ra_lw_physics": [4], "ra_sw_physics": [4], "bl_pbl_physics": [5],
                       "cu_physics": [0], "icloud": 3},
           "stoch": {"multi_perturb": [1], "pert_cld3": [True]}}
    errors = namelist_honour_report(nml, ("d01",))[0]
    assert any("pert_cld3" in e for e in errors) and any("multi_perturb" in e for e in errors)


def test_icloud_step_specs_are_registered():
    from gpuwrf.contracts.physics_interfaces import scheme_step_spec
    for value in (3, 2):
        spec = scheme_step_spec("radiation", 4, f"icloud={value}")
        assert spec.owner_module == "src/gpuwrf/physics/cloud_fraction.py"
        assert spec.wrf_slot == "first_rk_radiation_driver" and "CLDFRA" in spec.diagnostics
    assert "proofs/w41_icloud/oracle_gate.py" in scheme_step_spec("radiation", 4, "icloud=3").oracle
