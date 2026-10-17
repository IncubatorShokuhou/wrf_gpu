"""t-cfix-sfc (CALLER_AUDIT F3/F19/F13): sf_sfclay_physics=3/7/91 is ONE WRF surface call.

WRF surface_driver runs only the selected surface layer on phy_prep operands and
its grid INOUT memory; Noah-MP, the PBL (BL_GFS) and history consume that call
(module_surface_driver.F:2064-2077, 2165-2176, 2344-2353; pbl_driver.F:1400-1424).
The pristine numerical oracle is proofs/t_cfix_sfc (unmodified WRF 4.7.1 REAL4
public calls on real CPU-WRF c04 columns, ULP envelope); caller seams use the
operational step.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import sfclay_selected
from gpuwrf.coupling.sfclay_selected import column_call

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'proofs/t_cfix_sfc/fixtures/sfclay_wrf_oracle.npz'
META = json.loads((ROOT / 'proofs/t_cfix_sfc/fixtures/sfclay_wrf_oracle.json').read_text())
IN, OUT = META['inputs'], META['outputs']
WRF_OUTPUTS = {  # INOUT/OUT arguments of the original routines
    91: {'ust', 'znt', 'mol', 'zol', 'rmol', 'regime', 'psim', 'psih', 'fm', 'fh', 'qsfc', 'qgh', 'cpm', 'hfx',
         'qfx', 'lh', 'flhc', 'flqc', 'chs', 'chs2', 'cqs2', 'u10', 'v10', 'th2', 't2', 'q2', 'gz1oz0', 'wspd', 'br'},
    7: {'ust', 'znt', 'mol', 'zol', 'rmol', 'regime', 'psim', 'psih', 'qsfc', 'qgh', 'cpm', 'hfx', 'qfx', 'lh',
        'flhc', 'flqc', 'chs', 'chs2', 'cqs2', 'u10', 'v10', 'gz1oz0', 'wspd', 'br'},
    3: {'ust', 'znt', 'psim', 'psih', 'qsfc', 'qgh', 'cpm', 'hfx', 'qfx', 'lh', 'flhc', 'flqc', 'chs', 'chs2',
        'cqs2', 'u10', 'v10', 'gz1oz0', 'wspd', 'br'},
}
SLACK_ULP = 2


def _ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a)
    b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


@pytest.mark.parametrize('scheme', (91, 7, 3))
def test_selected_call_inside_pristine_wrf_envelope(scheme):
    d = np.load(FIXTURE)
    for call, inp in ((1, d['inputs']), (2, d[f'sf{scheme}_inp2'])):
        out = column_call(scheme, {k: jnp.asarray(inp[:, IN.index(k)]) for k in IN if k not in ('zol', 'regime')},
                          call)
        assert set(out) == WRF_OUTPUTS[scheme]
        wrf_regime = d[f'sf{scheme}_out{call}'][:, OUT.index('regime')]
        for key, value in out.items():
            value = np.asarray(value)
            assert value.dtype == np.float32 and np.isfinite(value).all(), key
            lo = d[f'sf{scheme}_lower{call}'][:, OUT.index(key)]
            hi = d[f'sf{scheme}_upper{call}'][:, OUT.index(key)]
            outside = np.maximum(_ulp(np.minimum(value, lo), lo), _ulp(np.maximum(value, hi), hi))
            if scheme == 91 and key == 'zol':
                # SFCLAY keeps the ENTRY ZOL in regime 1 (no port consumer holds ZOL).
                outside = np.where(wrf_regime == 1., 0, outside)
            assert outside.max() <= SLACK_ULP, (scheme, call, key, int(outside.max()), int(np.argmax(outside)))


def _case(sf, *, noahmp, bl=0):
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    # Distinct, physical grid memory: water + land, stable/unstable history.
    xland = jnp.asarray([[1., 2., 1.], [2., 1., 1.], [1., 1., 2.]])
    state = state.replace(xland=xland, mol=jnp.asarray([[.05, -.1, 0.], [.2, -.03, .01], [0., .1, -.2]]),
                          hfx=jnp.full((3, 3), 40.), qfx=jnp.full((3, 3), 2.e-5), pblh=jnp.full((3, 3), 600.),
                          qsfc=jnp.full((3, 3), .008), ustar=jnp.full((3, 3), .3), roughness_m=jnp.where(
                              xland > 1.5, 2.e-4, .1), mavail=jnp.full((3, 3), .6))
    nl = _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=bl, sf_sfclay_physics=sf,
                   use_noahmp=noahmp, sf_surface_physics=4 if noahmp else 0, ra_lw_physics=0, ra_sw_physics=0)
    return grid, state, nl


def _forbid_resolves(monkeypatch, om):
    import gpuwrf.physics.noahmp_coupler as coupler
    import gpuwrf.coupling.scan_adapters as adapters

    def forbidden(*args, **kwargs):
        raise AssertionError('a second surface-layer solve replaced the selected scheme')
    monkeypatch.setattr(coupler, 'surface_layer_with_diagnostics', forbidden)
    monkeypatch.setattr(adapters, 'surface_layer_with_diagnostics', forbidden)
    for key in (3, 7, 91):
        monkeypatch.setitem(om.SFCLAY_SCAN_ADAPTERS, key, forbidden)
    monkeypatch.setattr(om, 'surface_adapter', forbidden)


def _spy_selected(monkeypatch):
    calls = []
    real = sfclay_selected.selected_surface_step

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(sfclay_selected, 'selected_surface_step', spy)
    return calls


@pytest.mark.parametrize('sf', (3, 7, 91))
def test_noahmp_receives_the_selected_surface_call(monkeypatch, sf):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    _, state, nl = _case(sf, noahmp=True)
    _forbid_resolves(monkeypatch, om)
    calls = _spy_selected(monkeypatch)
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    carry = initial_operational_carry(state).replace(
        radiation_diagnostics=SimpleNamespace(swnorm=zero, glw=zero, coszen=zero),
        noahmp_land=SimpleNamespace(ch=zero + .01, cm=zero + .02))
    monkeypatch.setattr(om, '_noahmp_params', lambda *a: (None, None))
    monkeypatch.setattr(om, '_lane_noahmp_static', lambda *a: None)
    monkeypatch.setattr(om, '_noahmp_clock', lambda *a: None)
    seen = []

    def land(state_in, held, *args, **options):
        seen.append((state_in, options.get('surface_diagnostics')))
        return state_in, held
    monkeypatch.setattr(om, 'noahmp_surface_step', land)
    om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
    assert len(calls) == 1 and len(seen) == 1
    state_out, handoff, noah, out = calls[0]
    land_state, diagnostics = seen[0]
    assert diagnostics is noah
    # Noah-MP owns CHXY/CMXY: the land carry, not the sfclay CHS, seeds them.
    assert diagnostics.ch is carry.noahmp_land.ch and diagnostics.cm is carry.noahmp_land.cm
    for name, key in (('ustar', 'ust'), ('roughness_m', 'znt'), ('qsfc', 'qsfc'), ('hfx', 'hfx'), ('qfx', 'qfx')):
        np.testing.assert_array_equal(np.asarray(getattr(land_state, name), np.float32), out[key])
    np.testing.assert_array_equal(np.asarray(diagnostics.u10), out['u10'])
    np.testing.assert_array_equal(np.asarray(diagnostics.fluxes.ustar), out['ust'])


def _selected_inputs(monkeypatch):
    seen = []
    real = sfclay_selected.column_call

    def spy(sf, inputs, itimestep):
        out = real(sf, inputs, itimestep)
        seen.append((dict(inputs), out, itimestep))
        return out
    monkeypatch.setattr(sfclay_selected, 'column_call', spy)
    return seen


@pytest.mark.parametrize('sf', (3, 7, 91))
def test_two_step_caller_threads_wrf_memory_and_phy_prep_operands(monkeypatch, sf):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.coupling.physics_couplers import _dry_theta_view, _wrf_hydrostatic_pressure_from_state
    grid, state, nl = _case(sf, noahmp=False)
    _forbid_resolves(monkeypatch, om)
    seen = _selected_inputs(monkeypatch)
    carry = initial_operational_carry(state)
    first = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
    second_carry = first.carry.replace(state=first.state)
    om._physics_step_forcing(second_carry, nl, 20., run_radiation=False, first_timestep=False)
    assert len(seen) == 2
    (in1, out1, it1), (in2, out2, it2) = seen
    assert float(it1) == 1. and float(it2) == 2.
    # Step 1 operands are WRF phy_prep: DRY theta (not theta_m), hydrostatic P, PSFC = p8w(kts).
    import jax
    with jax.enable_x64(False):  # REAL phy_prep, as the qualified SF1 producer stages it
        theta_dry = np.asarray(_dry_theta_view(state), np.float32)[0]
        p_hyd, psfc = _wrf_hydrostatic_pressure_from_state(state, grid.metrics, output_dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(in1['th']), theta_dry)
    np.testing.assert_array_equal(np.asarray(in1['p']), np.asarray(p_hyd)[0])
    np.testing.assert_array_equal(np.asarray(in1['psfc']), np.asarray(psfc))
    assert not np.allclose(np.asarray(in1['th']), np.asarray(state.theta[0], np.float32))
    for key in ('ust', 'znt', 'mol', 'qsfc', 'hfx', 'qfx', 'pblh'):
        np.testing.assert_array_equal(np.asarray(in1[key]), np.asarray(getattr(state, {
            'ust': 'ustar', 'znt': 'roughness_m'}.get(key, key)), np.float32))
    # Step 2 consumes step 1's WRF INOUT memory (SF_GFS writes only UST/ZNT; others hold).
    for key in ('ust', 'znt', 'mol', 'qsfc', 'hfx', 'qfx'):
        expected = out1[key] if key in out1 else in1[key]
        np.testing.assert_array_equal(np.asarray(in2[key]), np.asarray(expected), err_msg=key)
    assert not np.array_equal(np.asarray(in2['ust']), np.asarray(in1['ust']))


def test_no_lsm_history_is_the_selected_call(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    for sf in (91, 3):
        _, state, nl = _case(sf, noahmp=False)
        _forbid_resolves(monkeypatch, om)
        calls = _spy_selected(monkeypatch)
        held = seed_history_diagnostics(state)._replace(t2=jnp.full((3, 3), 271.5, jnp.float32),
                                                        q2=jnp.full((3, 3), .003, jnp.float32))
        carry = initial_operational_carry(state).replace(history_diagnostics=held)
        result = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
        noah = calls[-1][2]
        hist = result.carry.history_diagnostics
        for key in ('u10', 'v10', 'lh', 'psfc'):
            np.testing.assert_array_equal(np.asarray(getattr(hist, key)), np.asarray(getattr(noah, key)))
        if sf == 91:
            np.testing.assert_array_equal(np.asarray(hist.t2), np.asarray(noah.t2))
        else:
            # SF_GFS has no T2/Q2 argument: without an LSM WRF history keeps them.
            np.testing.assert_array_equal(np.asarray(hist.t2), np.asarray(held.t2))
            np.testing.assert_array_equal(np.asarray(hist.q2), np.asarray(held.q2))


def test_gfs_pbl_consumes_the_selected_sf_gfs_solve(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    _, state, nl = _case(3, noahmp=False, bl=3)
    _forbid_resolves(monkeypatch, om)
    calls = _spy_selected(monkeypatch)
    seen = []

    def pbl(st, dt, grid, **kwargs):
        seen.append(kwargs)
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        rates = {key: zero for key in ('u', 'v', 'theta', 'qv')}
        return (st, rates) if kwargs.get('_with_sources') else st
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 3, pbl)
    om._physics_step_forcing(initial_operational_carry(state), nl, 0., run_radiation=False, first_timestep=True)
    assert len(calls) == 1 and len(seen) == 1
    handoff, out = calls[0][1], calls[0][3]
    forcing = seen[0].get('surface_forcing')
    assert forcing is handoff and seen[0].get('_with_sources')
    for key in ('psim', 'psih', 'br', 'wspd', 'gz1oz0', 'u10', 'v10'):
        np.testing.assert_array_equal(np.asarray(forcing[key]), np.asarray(out[key]))
    # BL_GFS reads TSKOLD = TSK at PBL entry (post-LSM), not a pre-LSM copy.
    assert 'tsk' not in forcing


@pytest.mark.parametrize('sf,ok', ((3, True), (1, False), (5, False), (91, False), (7, False)))
def test_gfs_pbl_admission_is_wrf_isfc2(sf, ok):
    from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection, resolve_physics_suite
    from gpuwrf.io.namelist_binding import namelist_honour_report
    config = {'bl_pbl_physics': 3, 'sf_sfclay_physics': sf}
    if ok:
        assert resolve_physics_suite(config).pbl.option == 3
    else:
        with pytest.raises(UnsupportedSchemeSelection, match='GFS PBL pairing'):
            resolve_physics_suite(config)
    errors, _ = namelist_honour_report({'physics': {k: [v] for k, v in config.items()}}, ('d01',))
    assert any('bl_pbl_physics=3' in e and 'GFS surface layer' in e for e in errors) is (not ok)
