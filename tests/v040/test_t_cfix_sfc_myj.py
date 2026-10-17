"""t-cfix-sfc (CALLER_AUDIT F3 sf=2 / F12 / F19): MYJSFC -> land -> MYJPBL is one coupled chain.

WRF runs MYJSFC once per step with carried INOUT memory (THZ0 QZ0 UZ0 VZ0 AKHS AKMS;
NTSD==1 cold start), Noah-MP consumes that call, and MYJPBL reads THIS solve's
exchange memory, ELFLX = post-LSM grid LH and CHKLOWQ = 1 (module_surface_driver.F:
2210-2224, 3429; module_pbl_driver.F:1439-1465), returning THZ0/QZ0/QSFC.  Numerical
oracle: proofs/t_cfix_sfc (unmodified WRF 4.7.1 MYJSFC REAL4 public call on real
CPU-WRF c03 profiles, 3 chained calls, +-1 ULP input envelope).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.sf_myj import myjsfc_columns

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'proofs/t_cfix_sfc/fixtures/myjsfc_wrf_oracle.npz'
META = json.loads((ROOT / 'proofs/t_cfix_sfc/fixtures/myjsfc_wrf_oracle.json').read_text())
PROF, SFC, OUT = META['prof'], META['sfc'], META['outputs']
PORT = {'ust': 'ustar', 'lh': 'flx_lh'}
# WRF's 5-iteration SFCDIF fixed point amplifies XLA-vs-glibc logf/expf rounding: the compiled
# REAL path stays inside the +-1 ULP input envelope up to 8 output ULPs (0.16 % of cells, disclosed).
SLACK_ULP = 8


def _ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a)
    b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


def _port(prof, sfc, itimestep):
    nz = prof.shape[1] - 1
    col = lambda k: jnp.asarray(prof[:, :nz, PROF.index(k)])
    s = lambda k: jnp.asarray(sfc[:, SFC.index(k)])
    return myjsfc_columns(col('u'), col('v'), col('t'), col('th'), col('qv'), col('qc'), col('pmid'), col('dz'),
                          col('q2'), tsk=s('tsk'), xland=s('xland'), z0base=s('z0base'),
                          psfc=jnp.asarray(prof[:, 0, PROF.index('pint')]), znt=s('znt'), ustar=s('ust'),
                          mavail=s('mavail'), qsfc=s('qsfc'), thz0=s('thz0'), qz0=s('qz0'), uz0=s('uz0'),
                          vz0=s('vz0'), itimestep=itimestep, akhs=s('akhs'), akms=s('akms'), wrf_real=True,
                          ht=s('ht'))


def test_myjsfc_three_chained_calls_inside_pristine_wrf_envelope():
    d = np.load(FIXTURE)
    checked = 0
    for call in (1, 2, 3):
        out = _port(d['prof'], d[f'sfc{call}'], call)
        for j, key in enumerate(OUT):
            if key == 'spare':
                continue
            value = np.asarray(out[PORT.get(key, key)], np.float32)
            assert np.isfinite(value).all(), key
            lo, hi = d[f'lower{call}'][:, j], d[f'upper{call}'][:, j]
            outside = np.maximum(_ulp(np.minimum(value, lo), lo), _ulp(np.maximum(value, hi), hi))
            assert outside.max() <= SLACK_ULP, (call, key, int(outside.max()), int(np.argmax(outside)))
            checked += 1
    assert checked == 3 * 35


def _case(*, noahmp, cu=0):
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    from gpuwrf.coupling.myj_surface import initial_myj_memory
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    xland = jnp.asarray([[1., 2., 1.], [2., 1., 1.], [1., 1., 2.]])
    state = state.replace(xland=xland, qke=jnp.full(state.theta.shape, .4), qsfc=jnp.full((3, 3), .008),
                          ustar=jnp.full((3, 3), .3), roughness_m=jnp.where(xland > 1.5, 2.e-4, .1),
                          mavail=jnp.full((3, 3), .6))
    nl = _namelist(grid, mp_physics=0, cu_physics=cu, bl_pbl_physics=2, sf_sfclay_physics=2,
                   use_noahmp=noahmp, sf_surface_physics=4 if noahmp else 0, ra_lw_physics=0, ra_sw_physics=0)
    held = initial_myj_memory(state)
    return grid, state, nl, held


def _carry(state, held):
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    return initial_operational_carry(state).replace(qnse=held, history_diagnostics=seed_history_diagnostics(
        state, myjsfc=True))


def _forbid_resolves(monkeypatch):
    import gpuwrf.physics.myj_adapters as adapters
    import gpuwrf.physics.noahmp_coupler as coupler

    def forbidden(*args, **kwargs):
        raise AssertionError('a second surface-layer solve replaced this step\'s MYJSFC call')
    monkeypatch.setattr(adapters, '_run_janjic', forbidden)
    monkeypatch.setattr(coupler, 'surface_layer_with_diagnostics', forbidden)


def _spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result
    monkeypatch.setattr(module, name, spy)
    return calls


def test_noahmp_and_myjpbl_consume_this_steps_myjsfc(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.coupling import myj_surface
    import gpuwrf.physics.myj_adapters as adapters
    _, state, nl, held = _case(noahmp=True)
    _forbid_resolves(monkeypatch)
    surface_calls = _spy(monkeypatch, myj_surface, 'myj_surface_step')
    columns = _spy(monkeypatch, adapters, 'myj_columns')
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    land = SimpleNamespace(ch=zero + .01, cm=zero + .02, sneqv=jnp.asarray([[0., 0., 5.], [0., 2., 0.], [0., 0., 0.]]))
    carry = _carry(state, held).replace(radiation_diagnostics=SimpleNamespace(swnorm=zero, glw=zero, coszen=zero),
                                        noahmp_land=land)
    monkeypatch.setattr(om, '_noahmp_params', lambda *a: (None, None))
    monkeypatch.setattr(om, '_lane_noahmp_static', lambda *a: None)
    monkeypatch.setattr(om, '_noahmp_clock', lambda *a: None)
    lsm_lh = jnp.asarray([[123., 50., 77.], [9., 300., 1.], [44., 5., 60.]], jnp.float32)
    seen = []

    def lsm(state_in, held_land, *args, **options):
        seen.append(options.get('surface_diagnostics'))
        names = ('t2', 'q2', 'u10', 'v10', 'lh', 'hfx', 'psfc')
        fields = {k: (lsm_lh if k == 'lh' else jnp.full((3, 3), 1., jnp.float32)) for k in names}
        return state_in, held_land, fields
    monkeypatch.setattr(om, 'noahmp_surface_step', lsm)
    result = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
    assert len(surface_calls) == 1 and len(seen) == 1 and len(columns) == 1
    _, held1, noah, out = surface_calls[0][2]
    assert seen[0] is noah and noah.ch is land.ch
    kw = columns[0][1]
    flat = lambda a: np.asarray(a, np.float64).reshape(-1)
    for key in ('akhs', 'akms', 'thz0', 'qz0', 'uz0', 'vz0', 'ct'):
        np.testing.assert_array_equal(np.asarray(kw[key]), flat(getattr(held1, key)), err_msg=key)
    np.testing.assert_array_equal(np.asarray(kw['ustar']), flat(out['ustar']))
    # ELFLX = post-LSM grid LH (pbl_driver.F:1457) and CHKLOWQ = 1 after Noah-MP (surface_driver.F:3429).
    np.testing.assert_array_equal(np.asarray(kw['elflx']), flat(lsm_lh))
    np.testing.assert_array_equal(np.asarray(kw['chklowq']), np.ones(9))
    np.testing.assert_array_equal(np.asarray(kw['snow']), flat(land.sneqv))
    # The PBL's returned THZ0/QZ0/QSFC are the memory the next MYJSFC call reads.
    pbl_out = columns[0][2]
    held2 = result.carry.qnse
    np.testing.assert_array_equal(np.asarray(held2.thz0).reshape(-1), np.asarray(pbl_out['THZ0'], np.float32))
    np.testing.assert_array_equal(np.asarray(held2.qz0).reshape(-1), np.asarray(pbl_out['QZ0'], np.float32))
    np.testing.assert_array_equal(np.asarray(result.state.qsfc, np.float32).reshape(-1),
                                  np.asarray(pbl_out['QSFC'], np.float32))
    np.testing.assert_array_equal(np.asarray(result.carry.history_diagnostics.u10e), np.asarray(out['u10e']))


def test_two_steps_thread_myjsfc_memory(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    import gpuwrf.physics.myj_adapters as adapters
    from gpuwrf.physics import sf_myj
    _, state, nl, held = _case(noahmp=False)
    _forbid_resolves(monkeypatch)
    calls = _spy(monkeypatch, sf_myj, 'myjsfc_columns')
    pbl = _spy(monkeypatch, adapters, 'myj_columns')
    carry = _carry(state, held)
    first = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
    om._physics_step_forcing(first.carry.replace(state=first.state), nl, 20., run_radiation=False,
                             first_timestep=False)
    assert len(calls) == 2 and len(pbl) == 2
    (_, kw1, out1), (_, kw2, _) = calls
    assert float(np.asarray(kw1['itimestep']).reshape(-1)[0]) == 1.
    assert float(np.asarray(kw2['itimestep']).reshape(-1)[0]) == 2.
    # TKE operand: WRF passes TKE_PBL (= qke/2) and MYJSFC doubles it itself.
    np.testing.assert_allclose(np.asarray(calls[0][0][8]), .5 * np.full((9, 12), .4), rtol=1e-6)
    p1 = pbl[0][2]
    for key in ('akhs', 'akms', 'uz0', 'vz0'):
        np.testing.assert_array_equal(np.asarray(kw2[key]), np.asarray(out1[key], np.float32).astype(np.float64),
                                      err_msg=key)
    # MYJPBL's THZ0/QZ0/QSFC (not the surface call's) are what step 2 reads.
    np.testing.assert_array_equal(np.asarray(kw2['thz0']), np.asarray(p1['THZ0'], np.float32).astype(np.float64))
    np.testing.assert_array_equal(np.asarray(kw2['qz0']), np.asarray(p1['QZ0'], np.float32).astype(np.float64))
    np.testing.assert_array_equal(np.asarray(kw2['qsfc']), np.asarray(p1['QSFC'], np.float32).astype(np.float64))
    # Without an LSM, ELFLX is MYJSFC's own LH; CHKLOWQ follows WRF's unset i1 field (zero policy).
    np.testing.assert_array_equal(np.asarray(pbl[0][1]['elflx']), np.asarray(out1['flx_lh'], np.float32).astype(np.float64))
    np.testing.assert_array_equal(np.asarray(pbl[0][1]['chklowq']), np.zeros(9))


def test_myj_requires_its_held_surface_memory():
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    _, state, nl, _ = _case(noahmp=False)
    with pytest.raises(ValueError, match='MYJSFC surface memory is missing'):
        om._physics_step_forcing(initial_operational_carry(state), nl, 0., run_radiation=False, first_timestep=True)


def test_initial_carry_seeds_registry_zero_myj_memory():
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    _, state, nl, _ = _case(noahmp=False)
    held = _initial_carry_for_run(state, nl).qnse
    assert held is not None
    for key in ('akhs', 'akms', 'thz0', 'qz0', 'uz0', 'vz0', 'ct', 'lh'):
        value = np.asarray(getattr(held, key))
        assert value.dtype == np.float32 and value.shape == (3, 3) and not value.any(), key


@pytest.mark.parametrize('format', ['pickle', 'netcdf', 'native'])
def test_myj_memory_restart_roundtrip_and_refusal(cpu_pallas_interpret, tmp_path, format):
    from test_restart_history_activity import _myj, _write_read
    grid, nml, carry, identity = _myj()
    assert carry.qnse is not None
    (tmp_path / 'ok').mkdir(); (tmp_path / 'missing').mkdir()
    restored = _write_read(tmp_path / 'ok', format, grid, nml, carry, identity)
    for a, b in zip(jax.tree.leaves(carry.qnse), jax.tree.leaves(restored.qnse), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    with pytest.raises(ValueError, match='MYJSFC.*E78'):
        _write_read(tmp_path / 'missing', format, grid, nml, carry.replace(qnse=None), identity)
