"""Morrison mp=10/40 REFL_10CM (and mp=40 MSKF_REFL_10CM) from WRF refl10cm_hm.

Oracle: pristine WRF 4.7.1 module_mp_morr_two_moment[_aero].F refl10cm_hm after
MORR_TWO_MOMENT_INIT[_AERO](0) (proofs/t_classd/morr_radar_oracle), REAL4 and REAL8 builds,
304 columns (300 real c08/c20 history columns incl. 150 melting-layer columns, 4 synthetic).
REAL4 BYTE on every level (incl. the DOUBLE melting integral); the radar permittivities follow
WRF's literal kinds (single-precision csqrt for water).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'proofs/t_classd/morr_radar_oracle/fixtures'


def _data():
    cols = np.load(FIX / 'inputs.npz')['columns']
    ref = np.load(FIX / 'oracle.npz')
    return cols, {k: ref[k] for k in ref.files}


def _port(cols, fn=None):
    from gpuwrf.diagnostics.morrison_radar import morrison_reflectivity
    fn = fn or morrison_reflectivity
    # Fresh closure: jax.jit caches traces per function object (FINDINGS E140/E161).
    return np.asarray(jax.jit(lambda *a: fn(*a))(*[cols[:, i, :] for i in range(9)]))


def _melting_levels(cols):
    t, qr, qs, qg = cols[:, 7], cols[:, 1], cols[:, 3], cols[:, 5]
    nz = t.shape[-1]
    k = np.arange(nz)
    trigger = (k[None, :-1] >= 0) & (t[:, :-1] > np.float32(273.15)) & (qr[:, :-1] > np.float32(1e-9)) & (
        (qs[:, 1:] > np.float32(1e-9)) | (qg[:, 1:] > np.float32(1e-9)))
    k0 = np.where(trigger.any(1), np.max(np.where(trigger, k[None, :-1] + 1, 0), axis=1), 0)
    return (k[None, :] < k0[:, None]) & (k0[:, None] >= 1)


def test_init_constants_match_pristine_radar_init():
    from gpuwrf.diagnostics.morrison_radar import CONSTANTS as C
    manifest = json.loads((FIX / 'manifest.json').read_text())
    for arm in ('fp32_morr', 'fp32_morr_aero'):
        k = manifest['arms'][arm]['constants']
        f32 = lambda values: np.asarray(values, np.float32)
        for key, sp in (('XCRG', 'r'), ('XCSG', 's'), ('XCGG', 'g')):
            assert f32(k[key]).tobytes() == np.asarray(C['c'][sp]['cg'], np.float32).tobytes(), key
        for key, sp in (('XCRE', 'r'), ('XCSE', 's'), ('XCGE', 'g')):
            assert f32(k[key]).tobytes() == np.asarray(C['c'][sp]['ce'], np.float32).tobytes(), key
        scal = f32(k['SCAL'])
        mine = [C['c']['r'][n] for n in ('am', 'bm', 'mu', 'obm')] + [C['c']['s'][n] for n in ('am', 'bm', 'obm', 'ocm')] + \
               [C['c']['g'][n] for n in ('am', 'bm', 'obm', 'ocm')]
        assert scal.tobytes() == np.asarray(mine, np.float32).tobytes()
        assert f32(k['XORG']).tobytes() == np.asarray([C['c'][s]['og2'] for s in 'rsg'], np.float32).tobytes()
        kw, pi5, lamda4 = k['KPL']
        assert pi5 == C['pi5'] and lamda4 == C['lamda4']
        assert abs(kw - C['kw']) <= 4 * np.finfo(np.float64).eps * kw
        mw_re, mw_im, mi_re, mi_im = k['MWI']
        assert (C['m_w'].real, C['m_w'].imag) == (mw_re, mw_im)  # single-precision csqrt, BYTE
        assert abs(complex(mi_re, mi_im) - C['m_i']) <= 8 * np.finfo(np.float64).eps * abs(C['m_i'])


def test_refl10cm_hm_matches_pristine_real4():
    cols, ref = _data()
    assert ref['fp32_morr'].tobytes() == ref['fp32_morr_aero'].tobytes()  # identical WRF bodies
    got = _port(cols)
    want = ref['fp32_morr']
    assert got.dtype == np.float32 and np.isfinite(got).all() and np.isfinite(want).all()
    assert got.tobytes() == want.tobytes(), np.argwhere(got != want)[:10]
    # Independent scale: WRF's own REAL4-vs-REAL8 spread on melting levels is ~3e-5 dBZ.
    spread = np.abs(ref['fp32_morr'].astype(np.float64) - ref['fp64_morr'])
    assert spread[_melting_levels(cols) & (want > -35)].max() > 1e-6


def test_oracle_census_covers_every_branch():
    """E154: rain, snow, graupel, mixed and melting (wet snow / wet graupel) levels are active."""
    cols, ref = _data()
    want = ref['fp32_morr']
    act = lambda i: cols[:, i] > np.float32(1e-9)
    live = want > -35.
    counts = dict(rain=np.count_nonzero(act(1) & live), snow=np.count_nonzero(act(3) & live),
                  graupel=np.count_nonzero(act(5) & live), melting=np.count_nonzero(_melting_levels(cols) & live),
                  melting_graupel=np.count_nonzero(_melting_levels(cols) & act(5) & live))
    assert all(v > 20 for v in counts.values()), counts
    assert np.count_nonzero(want <= -35.) > 1000  # no-hydrometeor floor (raw -35.23 dBZ) exercised


@pytest.mark.parametrize('mutant', ['graupel_density_900', 'no_melting', 'rain_threshold'])
def test_mutants_break_parity(mutant, monkeypatch):
    import gpuwrf.diagnostics.morrison_radar as M
    cols, ref = _data()
    if mutant == 'graupel_density_900':  # IHAIL=1 constants instead of morr_rimed_ice=0 graupel
        c = dict(M.CONSTANTS); c['c'] = dict(c['c'])
        g = dict(c['c']['g']); g['am'] = np.float32(np.float32(np.float32(900.) * M.PI) / np.float32(6.)); c['c']['g'] = g
        monkeypatch.setattr(M, 'CONSTANTS', c)
    elif mutant == 'no_melting':
        monkeypatch.setattr(M.lax, 'cond', lambda pred, a, b, *ops: b(*ops))
    else:
        monkeypatch.setattr(M, '_species', (lambda orig: lambda q, n, rho, p, thr: orig(q, n, rho, p, np.float32(1e-8)))(M._species))
    got = _port(cols, M.morrison_reflectivity)
    assert np.count_nonzero(got != ref['fp32_morr']) > 0


def test_caller_clamp_and_history_cadence():
    from gpuwrf.diagnostics.morrison_radar import morrison_refl_10cm
    from gpuwrf.diagnostics.online import radar_from_context
    cols, ref = _data()
    ny, nx, nz = 2, 3, cols.shape[2]
    context = tuple(jnp.asarray(cols[:ny * nx, i, :]) for i in range(9))
    expected = np.maximum(np.float32(-35.), ref['fp32_morr'][:ny * nx])
    np.testing.assert_array_equal(np.asarray(jax.jit(morrison_refl_10cm)(*context)), np.asarray(_port(cols)[:ny * nx]).clip(-35.))
    previous = np.full((nz, ny, nx), 99., np.float32)
    for mp in (10, 40):
        nl = SimpleNamespace(mp_physics=mp, do_radar_ref=1, nwp_diagnostics=0, run_physics=True,
                             dt_s=54., diagnostic_history_interval=60.)
        call = jax.jit(lambda step: radar_from_context(jnp.asarray(previous), context, nl, step, 0))
        for step in (1, 66, 67, 68):
            got = np.asarray(call(jnp.int32(step)))
            if step == 67:  # 67*54 s crosses the 60 min history alarm
                full = np.moveaxis(expected.reshape(ny, nx, nz), -1, 0)
                assert got.tobytes() == full.tobytes()
            else:
                np.testing.assert_array_equal(got, previous)


def test_admission_radar_only_and_nwp_refused():
    from gpuwrf.diagnostics.online import configuration, validate_options
    from gpuwrf.io.namelist_binding import NamelistNotHonouredError, require_namelist_honoured
    for mp in (10, 40):
        nl = SimpleNamespace(mp_physics=mp, do_radar_ref=1, nwp_diagnostics=0, run_physics=True, dt_s=54.,
                             diagnostic_history_interval=60.)
        validate_options(nl, specified=True, nested=False, periodic=False)
        assert configuration(SimpleNamespace(mp_physics=mp, do_radar_ref=0, nwp_diagnostics=0))['do_radar_ref'] == 0
        with pytest.raises(ValueError, match='NWP'):
            validate_options(SimpleNamespace(**{**vars(nl), 'nwp_diagnostics': 1}), specified=True, nested=False,
                             periodic=False)
        require_namelist_honoured({'physics': {'mp_physics': [mp], 'cu_physics': [0], 'do_radar_ref': [1]}}, ('d01',))
        with pytest.raises(NamelistNotHonouredError):
            require_namelist_honoured({'physics': {'mp_physics': [mp], 'cu_physics': [0], 'do_radar_ref': [1]},
                                       'time_control': {'nwp_diagnostics': 1}}, ('d01',))


@pytest.mark.parametrize('mp', [10, 40])
def test_adapter_radar_context_is_the_mp_exit_state(mp, monkeypatch):
    """Regression-only seam check: radar selection must not change the microphysics result."""
    import importlib.util
    from gpuwrf.contracts import state as sc
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS, _mp_in
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])
    spec = importlib.util.spec_from_file_location('classd_lw', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    g = f._grid(nz=8, ny=3, nx=4)
    st = f._state(g).ensure_conditional_leaves(mp_physics=mp)
    adapter = MP_SCAN_ADAPTERS[mp]
    plain = adapter(st, 18., g)
    with_radar, context = adapter(st, 18., g, return_radar=True)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(with_radar)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    ny, nx, nz = 3, 4, 8
    for value, leaf in zip(context[:7], ('qv', 'qr', 'Nr', 'qs', 'Ns', 'qg', 'Ng')):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(_mp_in(getattr(with_radar, leaf), ny, nx, nz)))
    np.testing.assert_array_equal(np.asarray(context[8]), np.asarray(_mp_in(st.p, ny, nx, nz)))
    assert context[7].shape == (ny * nx, nz) and np.isfinite(np.asarray(context[7])).all()


@pytest.mark.parametrize('mp', [10, 40])
def test_operational_seed_and_post_rk_context(mp, monkeypatch):
    from dataclasses import replace
    import importlib.util
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime import operational_mode as op
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])
    spec = importlib.util.spec_from_file_location('classd_lw2', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    g = f._grid(nz=8, ny=5, nx=6)
    st = f._state(g).ensure_conditional_leaves(mp_physics=mp)
    nl = replace(f._namelist(g), mp_physics=mp, do_radar_ref=1)
    carry = op._initial_carry_for_run(st, nl)
    assert carry.radar_reflectivity is not None and carry.radar_reflectivity.shape == st.qv.shape
    updated, precip, context = op._apply_post_rk_microphysics(st, nl, return_precipitation=True,
                                                              return_radar_context=True, step_index=jnp.int32(1))
    assert len(context) == 9
    # Composition with the public precipitation carry (t-calleraudit F1): the radar call returns
    # the same RAINNCV/SNOWNCV/GRAUPELNCV slots as the radar-off call.
    plain, plain_precip = op._apply_post_rk_microphysics(st, nl, return_precipitation=True)
    assert set(precip) == set(plain_precip)
    for a, b in zip(jax.tree.leaves((plain, plain_precip)), jax.tree.leaves((updated, precip)), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize('radar', [0, 1])
def test_nested_writer_mskf_reflectivity(tmp_path, radar):
    """mp=40: MSKF_REFL_10CM = held REFL_10CM with do_radar_ref=1, WRF's untouched +0 otherwise."""
    from gpuwrf.io.netcdf_lock import Dataset
    from test_initial_history_wrf_t0 import _writer
    writer, state = _writer(tmp_path, lambda *a, **kw: {})
    nl = writer.bundles['d01'].namelist
    writer.bundles['d01'].namelist = {**nl, 'mp_physics': 40, 'do_radar_ref': radar, 'physics': {'mp_physics': 40}}
    cols, ref = _data()
    nz, ny, nx = state.theta.shape
    held = None
    if radar:
        held = np.broadcast_to(np.maximum(ref['fp32_morr'][0, :nz], np.float32(-35.))[:, None, None],
                               state.theta.shape).copy()
    result = writer('d01', 200, SimpleNamespace(state=state, radar_reflectivity=held, nwp_diagnostics=None))
    with Dataset(result['wrfout']) as ds:
        mskf = np.asarray(ds['MSKF_REFL_10CM'][0], np.float32)
        if radar:
            np.testing.assert_array_equal(np.asarray(ds['REFL_10CM'][0]), held)
            assert mskf.tobytes() == held.tobytes()
        else:
            assert np.all(mskf.view(np.uint32) == 0) and 'REFL_10CM' not in ds.variables


@pytest.mark.parametrize('mp', [10, 40])
def test_changed_morrison_radar_constants_change_aot_keys(mp, tmp_path, monkeypatch):
    """t-aotkey rule: refl10cm_hm constants are host globals baked into the program -> they key it."""
    import gpuwrf.diagnostics.morrison_radar as M
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime import aot_cheap_key as ck, aot_executable as ax, aot_precompile as ap
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=mp, do_radar_ref=1,
                             ra_sw_physics=0, ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0)
    off = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=mp, do_radar_ref=0,
                              ra_sw_physics=0, ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0)

    @jax.jit
    def selected(x, nl):
        return x + jnp.float32(M.CONSTANTS['c']['s']['am'])

    x = jnp.zeros(2, jnp.float32)
    before, before_off = baked_table_hash(selected, nl), baked_table_hash(selected, off)
    key = ck.cheap_key(selected, (x, nl), {}, nl)
    assert key is not None
    lowered = selected.lower(x, nl)
    assert ap._serialize_domain_blob('morrison_radar', lowered.compile(), tmp_path, lowered=lowered,
                                     cheap_key=key, key_schema=ck.KEY_SCHEMA)['aot_written']
    changed = dict(M.CONSTANTS)
    changed['c'] = {**M.CONSTANTS['c'], 's': {**M.CONSTANTS['c']['s'],
                                              'am': np.float32(M.CONSTANTS['c']['s']['am'] * np.float32(1.01))}}
    monkeypatch.setattr(M, 'CONSTANTS', changed)
    selected.clear_cache()
    assert baked_table_hash(selected, nl) != before
    assert baked_table_hash(selected, off) == before_off  # radar off: constants are not traced
    after = ck.cheap_key(selected, (x, nl), {}, nl)
    assert after is not None and after != key
    loaded, _status = ap.load_domain_blob('morrison_radar', tmp_path, cheap_key=after, return_status=True)
    assert loaded is None
    assert ax.hlo_sha256_from_lowered(selected.lower(x, nl)) != ax.hlo_sha256_from_lowered(lowered)
    selected.clear_cache()
