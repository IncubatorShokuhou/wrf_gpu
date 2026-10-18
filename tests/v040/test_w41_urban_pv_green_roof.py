"""BEP+BEM rooftop PV (PV_FRAC_ROOF>0) and green roofs (GR_FLAG=1): pristine oracle + writer gates.

Reference: unmodified pristine WRF 4.7.1 BEP/BEM modules driven by the qualified v040 caller
(proofs/w41_writergaps/pvgr_oracle.F90 = proofs/v040/urban_bep_bem/bep_bem_oracle.F90 + history
dumps), six regimes x six dt=6 s steps per arm, REAL4 and REAL8 builds.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT/'proofs/w41_writergaps'
SAVE = PROOF/'savepoints'
sys.path.insert(0, str(PROOF))
sys.path.insert(0, str(ROOT/'proofs/v040/urban_bep_bem'))
from gpuwrf.config import paths as _paths  # noqa: E402

URBAN = ('QGR_URB4D', 'TGR_URB4D', 'DRAINGR_URB3D', 'EP_PV_URB3D')
ARMS_PVGR = ('pv04_ac_on', 'pv10_ac_off', 'gr05_ac_on', 'gr10pv03_ac_on', 'gr05irr_ac_off', 'gr05dt_ac_on',
             'gr05t1_ac_on', 'gr05dry_ac_on')
ARMS_REAL4 = ('pv04_ac_on', 'gr10pv03_ac_on', 'gr05irr_ac_off', 'gr05dt_ac_on', 'gr05t1_ac_on', 'gr05dry_ac_on')
# Fields the harness compares per step: 54 roof/canyon/building/surface, +7 green-roof, +1 with green roof and PV.
COMPARED = {'pv04_ac_on': 54, 'pv10_ac_off': 54, 'gr05_ac_on': 61, 'gr10pv03_ac_on': 62, 'gr05irr_ac_off': 61, 'gr05dt_ac_on': 61,
            'gr05t1_ac_on': 61, 'gr05dry_ac_on': 61}
# REAL4 history producers that stay inside WRF's own REAL4-vs-REAL8 envelope in every arm/step.
ENVELOPE_FIELDS = ('EP_PV_URB3D', 'T_PV_URB3D', 'QGR_URB3D', 'TGR_URB3D', 'TRV_URB4D', 'DRAIN_URB4D')
# Frozen REAL4 max-abs caps (this oracle, all arms/steps) for the conditioned green-roof fluxes;
# the v040 ordinary-roof fluxes (SFR/SFW/LFR/...) keep their own ROUND01 qualification.
FROZEN_ABS = {'SFRV_URB3D': 0.05, 'LFRV_URB3D': 0.02, 'QR_URB4D': 3e-6, 'DRAINGR_URB3D': 2e-6}


def _require_pristine_tables():
    """Skip, naming the resolved directory, when the unmodified WRF run tables are not installed."""
    run = _paths.wrf_run_dir()
    missing = [name for name in ('URBPARM.TBL', 'MPTABLE.TBL') if not (run/name).is_file()]
    if missing:
        pytest.skip(f'pristine WRF run tables {missing} unavailable in {run} '
                    '(set GPUWRF_WRF_ROOT or link data/wrf_pristine)')


@pytest.fixture(autouse=True, scope='module')
def _pristine_wrf_root():
    """run_oracle.table reads build_oracle.WRF, a lane-host path: resolve the pristine tree like gpuwrf
    (GPUWRF_WRF_ROOT > <repo>/data/wrf_pristine/WRF) and skip table consumers when it is absent."""
    import run_oracle
    original = run_oracle.table

    def table(changes):
        _require_pristine_tables()
        return original(changes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_oracle, 'WRF', _paths.wrf_root())
        patch.setattr(run_oracle, 'table', table)
        yield


def _geo(arm, dtype=np.float32, tmp=None):
    from run_oracle import ARMS, table
    from gpuwrf.physics.urban_bep_static import read_bep_classes
    from gpuwrf.physics.urban_bem_static import bem_class
    path = Path(tmp)/f'{arm}.TBL'
    path.write_text(table(ARMS[arm]))
    _, classes = read_bep_classes(str(path), dtype)
    return bem_class(classes[1], 1, dtype), classes[1]


@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    from pvgr_parity import run
    work = tmp_path_factory.mktemp('pvgr')
    jax.config.update('jax_enable_x64', True)
    result = run(SAVE, work, None, list(ARMS_PVGR), 6, precisions=('fp64',))
    result.update(run(SAVE, work, None, list(ARMS_REAL4), 6, precisions=('fp32',)))
    return result


def test_oracle_fixtures_are_complete_and_control_reproduces_v040_savepoints():
    manifest = json.loads((SAVE/'manifest.json').read_text())
    from run_oracle import ARMS
    for arm in ARMS:
        for precision in ('fp64', 'fp32'):
            assert manifest[f'{arm}_{precision}.npz']['fields'] == 91
    # Lane build reproduces the committed qualified ordinary-roof fixtures; only the
    # constructed SMOIS_URB seed (QR_URB4D) differs.
    for ac in ('ac_on', 'ac_off'):
        for precision in ('fp32', 'fp64'):
            new = np.load(SAVE/f'ctl_{ac}_{precision}.npz');old = np.load(ROOT/f'proofs/v040/urban_bep_bem/savepoints/urban3_{precision}_{ac}.npz')
            differ = [k for k in old.files if not np.array_equal(new[k], old[k])]
            assert differ == ['QR_URB4D'], (ac, precision, differ)


def test_oracle_arms_exercise_the_branches():
    for arm, field in [('pv04_ac_on', 'EP_PV_URB3D'), ('gr05_ac_on', 'DRAINGR_URB3D'), ('gr05_ac_on', 'SFRV_URB3D'),
                       ('gr10pv03_ac_on', 'EP_PV_URB3D'), ('gr05irr_ac_off', 'DRAINGR_URB3D')]:
        data = np.load(SAVE/f'{arm}_fp64.npz')
        assert np.count_nonzero(data[field][:, 1:]) > 0, (arm, field)
    ctl = np.load(SAVE/'ctl_ac_on_fp64.npz');pv = np.load(SAVE/'pv04_ac_on_fp64.npz');gr = np.load(SAVE/'gr05_ac_on_fp64.npz')
    assert not np.array_equal(ctl['TRB_URB4D'], pv['TRB_URB4D']) and not np.array_equal(ctl['TRB_URB4D'], gr['TRB_URB4D'])
    assert not np.array_equal(ctl['QR_URB4D'][:, -1], gr['QR_URB4D'][:, -1])
    irr = np.load(SAVE/'gr05irr_ac_off_fp64.npz')
    assert not np.array_equal(irr['QR_URB4D'][:, -1], np.load(SAVE/'gr05_ac_on_fp64.npz')['QR_URB4D'][:, -1])
    # wall_gr coupling: the 4 K green-soil contrast must move the roof layers well beyond round-off.
    dt = np.load(SAVE/'gr05dt_ac_on_fp64.npz')
    assert np.all(dt['TRV_URB4D'][:, 0]-gr['TRV_URB4D'][:, 0] == 4.)
    assert np.max(np.abs(dt['TRB_URB4D'][:, -1]-gr['TRB_URB4D'][:, -1])) > .1
    # GR_TYPE=1 (grass: rsmin 40, LAI 2, albedo .3) vs the pristine table's sedum (2).
    t1 = np.load(SAVE/'gr05t1_ac_on_fp64.npz')
    assert np.max(np.abs(t1['LFRV_URB3D']-gr['LFRV_URB3D'])) > 10.
    # Dry soil: every flux_flat_roof layer 5..10 below the wilting point qw=.06 at every step and level, so
    # gx clamps to 0, f4=0 and rsveg=rsmax (module_sf_bep_bem.F:4557-4567); QR_URB4D (level, layer, direction).
    dry = np.load(SAVE/'gr05dry_ac_on_fp64.npz')
    qr = dry['QR_URB4D'].reshape(*dry['QR_URB4D'].shape[:2], 18, 10, 2)[:, :, :, 4:, :]
    assert qr.max() < .06 and qr.min() > 0.
    assert np.max(np.abs(dry['LFRV_URB3D'])) < .1*np.max(np.abs(gr['LFRV_URB3D']))


@pytest.mark.parametrize('arm', ARMS_PVGR)
def test_real8_port_matches_pristine_wrf_on_every_field_and_step(parity, arm):
    result = parity[f'{arm}_fp64']
    assert result['violations'] == 0, result['worst']
    assert result['field_steps'] == 6*COMPARED[arm]


@pytest.mark.parametrize('arm', ARMS_REAL4)
def test_real4_history_producers_inside_wrf_envelope(parity, arm):
    rows = parity[f'{arm}_fp32']['fields']
    for row in rows:
        if row['field'] in ENVELOPE_FIELDS:
            assert row['violations'] == 0, row
        if row['field'] in FROZEN_ABS:
            assert row['max_abs'] <= FROZEN_ABS[row['field']], row


def test_mutant_without_pv_sensible_share_fails_real8(tmp_path, monkeypatch):
    """Genuine mutant: drop sfrpv from buildings() -> REAL8 B_T_BEP violation (deletion sensitive)."""
    from pvgr_parity import run
    import gpuwrf.physics.urban_bem_building as B
    real = B.pv_panel_step

    def mutant(*a, **k):
        panel = real(*a, **k)
        return panel._replace(sensible=jnp.zeros_like(panel.sensible))
    monkeypatch.setattr(B, 'pv_panel_step', mutant)
    result = run(SAVE, tmp_path, None, ['pv04_ac_on'], 2, precisions=('fp64',))['pv04_ac_on_fp64']
    assert result['violations'] > 0 and 'B_T_BEP' in result['worst']


def test_mutant_without_wall_gr_coupling_fails_real8(tmp_path, monkeypatch):
    """Genuine mutant: roof conduction without the layer-5 green-roof flux (wall instead of wall_gr)."""
    from pvgr_parity import run
    import gpuwrf.physics.urban_green_roof as G
    real = G.roof_wall_gr
    monkeypatch.setattr(G, 'roof_wall_gr', lambda *a, hfgr, **k: real(*a, hfgr=jnp.zeros_like(hfgr), **k))
    # gr05dt: constructed 4 K green-soil/roof contrast (pvgr_oracle.F90 PVGR_TRV_OFFSET), so hfgr is O(10 W/m2);
    # in the TBLEND-seeded arms hfgr stays < 1e-11 W/m2 over six steps and the coupling is unexercised.
    result = run(SAVE, tmp_path, None, ['gr05dt_ac_on'], 2, precisions=('fp64',))['gr05dt_ac_on_fp64']
    assert result['violations'] > 0 and 'TRB_URB4D' in result['worst']


def test_mutant_grass_with_sedum_stomata_fails_real8(tmp_path, monkeypatch):
    """Genuine mutant: GR_TYPE=1 uses the type-2 stomatal constants -> REAL8 violation on the grass arm."""
    from pvgr_parity import run
    import gpuwrf.physics.urban_green_roof as G
    monkeypatch.setattr(G, 'STOMATA', {1: G.STOMATA[2], 2: G.STOMATA[2]})
    result = run(SAVE, tmp_path, None, ['gr05t1_ac_on'], 2, precisions=('fp64',))['gr05t1_ac_on_fp64']
    assert result['violations'] > 0 and 'LFRV_URB3D' in result['worst']


def test_mutant_dry_soil_without_rsmax_cap_fails_real8(tmp_path, monkeypatch):
    """Genuine mutant: no rsmax cap; dry soil (f4=0) puts rsveg exactly on that branch."""
    from pvgr_parity import run
    import gpuwrf.physics.urban_green_roof as G
    monkeypatch.setattr(G, 'STOMATA', {k: (v[0], 1e30, v[2]) for k, v in G.STOMATA.items()})
    result = run(SAVE, tmp_path, None, ['gr05dry_ac_on'], 2, precisions=('fp64',))['gr05dry_ac_on_fp64']
    assert result['violations'] > 0 and 'LFRV_URB3D' in result['worst']


def test_table_admission(tmp_path):
    from run_oracle import table
    from gpuwrf.physics.urban_bep_static import read_bep_classes
    from gpuwrf.physics.urban_bem_static import bem_class
    def geo(changes):
        path = tmp_path/'t.TBL';path.write_text(table(changes))
        _, classes = read_bep_classes(str(path), np.float32)
        return bem_class(classes[1], 1, np.float32)
    assert geo({'PV_FRAC_ROOF': '0.4, 0.4, 0.4'}).params['pv_frac_roof'] == np.float32(.4)
    g = geo({'GR_FLAG': '1', 'GR_FRAC_ROOF': '0.5, 0.5, 0.5'}).params
    assert (g['gr_flag'], g['gr_frac_roof'], g['gr_type']) == (1, np.float32(.5), 2) and len(g['irho']) == 24
    for bad in ({'GR_FRAC_ROOF': '0.5, 0.5, 0.5'}, {'GR_FLAG': '1'}, {'GR_FLAG': '2', 'GR_FRAC_ROOF': '0.5, 0.5, 0.5'}):
        with pytest.raises(NotImplementedError):
            geo(bad)
    from gpuwrf.physics.urban_green_roof import config_from_params
    with pytest.raises(NotImplementedError, match='GR_TYPE'):
        config_from_params(dict(g, gr_type=3))


def test_table_values_reach_the_aot_key(tmp_path):
    """t-aotkey rule: the urban bundle rides a content-hashed static holder; PV/GR table changes rekey."""
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    base = _geo('ctl_ac_on', tmp=tmp_path)[0]
    pv = _geo('pv04_ac_on', tmp=tmp_path)[0]
    gr = _geo('gr05_ac_on', tmp=tmp_path)[0]
    digests = {canonical_digest(x) for x in (base, pv, gr)}
    assert len(digests) == 3


def _hook_case(tmp_path, arm):
    from gpuwrf.coupling import urban_multilayer_hook as hook
    from run_oracle import ARMS, table
    import shutil
    from gpuwrf.config.paths import wrf_run_dir
    _require_pristine_tables()
    run_dir = tmp_path/arm;run_dir.mkdir()
    for name in ('URBPARM.TBL', 'MPTABLE.TBL'):
        shutil.copy(wrf_run_dir()/name, run_dir/name)
    (run_dir/'URBPARM.TBL').write_text(table(ARMS[arm]))
    ivg = np.full((4, 5), 10, np.int32);ivg[1, 2] = ivg[2, 3] = 13
    bundle = hook.load_multilayer_bundle(run_dir, option=3, ivgtyp=ivg, isurban=13)
    smois = np.stack([np.full((4, 5), .27, np.float32)]+[np.full((4, 5), .25, np.float32)]*3)
    carry = hook.initial_multilayer_carry(bundle, tslb=jnp.full((4, 4, 5), 290., jnp.float32),
                                          frc_urb2d=jnp.zeros((4, 5), jnp.float32), nz=8, smois=smois)
    return hook, bundle, carry


def test_initial_carry_seeds_wrf_urban_var_init_values(tmp_path):
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    ex = carry.state.extra
    from gpuwrf.physics.urban_bem_model import GR_EXTRA, PV_EXTRA
    assert set(ex) == set(GR_EXTRA)|set(PV_EXTRA)
    assert np.all(np.asarray(ex['ep_pv']) == 0) and np.all(np.asarray(ex['tgr']) == 0) and np.all(np.asarray(ex['draingr']) == 0)
    assert np.all(np.asarray(ex['gr_moisture']) == np.float32(.27)) and np.all(np.asarray(ex['qgr']) == np.float32(.27))
    assert np.all(np.asarray(ex['gr_temperature']) == np.float32(bundle.bem.params['tblend']))
    hook2, bundle2, carry2 = _hook_case(tmp_path, 'ctl_ac_on')
    assert carry2.state.extra == {}
    fields = hook.multilayer_history_fields(bundle, carry)
    assert set(fields) == set(URBAN)
    assert np.asarray(fields['QGR_URB4D']).tobytes() == np.full((4, 5), .27, np.float32).tobytes()
    assert set(hook2.multilayer_history_fields(bundle2, carry2)) == {'QGR_URB4D'}


def test_history_fields_scatter_compact_values_like_wrf(tmp_path):
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    n = len(bundle.indices)
    extra = dict(carry.state.extra, ep_pv=jnp.arange(1, n+1, dtype=jnp.float32), tgr=jnp.full((n,), 300., jnp.float32),
                 draingr=jnp.full((n,), 1e-3, jnp.float32))
    fields = hook.multilayer_history_fields(bundle, carry._replace(state=carry.state._replace(extra=extra)))
    ep = np.asarray(fields['EP_PV_URB3D']).reshape(-1)
    assert ep[list(bundle.indices)].tolist() == [1., 2.] and np.count_nonzero(ep) == 2
    assert np.count_nonzero(np.asarray(fields['TGR_URB4D'])) == 2


def test_carry_state_mismatch_is_refused_e78(tmp_path):
    from gpuwrf.physics import urban_bem_model as M
    data = dict(np.load(SAVE/'pv04_ac_on_fp32.npz'))
    from bem_driver_parity import unpack
    state, forcing, rural, frc, other = unpack(data, 'fp32')
    for arm, extra in [('pv04_ac_on', {}), ('gr05_ac_on', {}), ('ctl_ac_on', {'ep_pv': jnp.zeros(6)})]:
        geo = _geo(arm, tmp=tmp_path)[0]
        latitude, direct, diffuse, rain, qv = other
        with pytest.raises(ValueError, match='E78'):
            jax.eval_shape(lambda st: M.bem_column_step(st, forcing, geo, latitude=latitude, sw_direct=direct,
                                                        sw_diffuse=diffuse, rain=rain, qv=qv, fraction=frc),
                           state._replace(extra=extra))


def test_producer_history_admits_active_branches_only():
    from gpuwrf.io.producer_history import inactive_names, add_fields
    lookup = lambda obj, key, default=None: obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    def nl(pv=0., gr=0):
        return {'sf_urban_physics': 3, 'urban_bundle': SimpleNamespace(
            bem=SimpleNamespace(params={'pv_frac_roof': pv, 'gr_flag': gr, 'gr_frac_roof': .5 if gr else 0.}))}
    assert set(inactive_names(nl(), lookup)) >= {'TGR_URB4D', 'DRAINGR_URB3D', 'EP_PV_URB3D'}
    assert 'EP_PV_URB3D' not in inactive_names(nl(pv=.4), lookup)
    assert {'TGR_URB4D', 'DRAINGR_URB3D'}.isdisjoint(inactive_names(nl(gr=1), lookup))
    assert 'EP_PV_URB3D' in inactive_names(nl(gr=1), lookup)


@pytest.mark.parametrize('full', (False, True))
def test_writer_emits_computed_fields_with_cpu_metadata(tmp_path, full):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    sys.path.insert(0, str(ROOT/'tests'))
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    cpu = json.loads((ROOT/'proofs/t_classd/URBAN_CPU_ALL_FRAMES.json').read_text())['frames'][1]['fields']
    state, grid, nlist = synthetic_case()
    nlist.update(sf_urban_physics=3, urban_bundle=SimpleNamespace(
        bem=SimpleNamespace(params={'pv_frac_roof': np.float32(.3), 'gr_flag': 1, 'gr_frac_roof': np.float32(1.)})))
    values = {name: (np.arange(20, dtype=np.float32).reshape(4, 5)+i)/np.float32(7) for i, name in enumerate(URBAN)}
    auth = writer_authority(grid, 'd01')
    p = prepare_wrfout_payload(state, grid, nlist, tmp_path/'urban', domain='d01', domain_authority=auth,
                               valid_time='2026-10-13T01:00:00', run_start='2026-10-13T00:00:00', lead_hours=1.,
                               full_variable_set=full, variable_subset=URBAN, diagnostics=values)
    write_prepared_wrfout(p, expected_domain='d01', expected_domain_authority=auth, variable_subset=URBAN)
    with Dataset(p.target) as ds:
        for name in URBAN:
            v = ds[name];ref = cpu[name]
            assert list(v.dimensions) == ref['dims'] and str(v.dtype) == ref['dtype']
            assert v.units == ref['units'] and v.stagger == ref['stagger']
            assert np.asarray(v[0], np.float32).tobytes() == values[name].tobytes()


def test_restart_schema_and_pickle_roundtrip_keep_pv_green_state(tmp_path):
    import pickle
    from gpuwrf.io.restart_schema import validate_urban_bep
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    validate_urban_bep(carry, {'sf_urban_physics': 3}, grid_shape=(4, 5), nz=8)
    back = pickle.loads(pickle.dumps(jax.tree.map(np.asarray, carry)))
    assert set(back.state.extra) == set(carry.state.extra)
    for key, value in carry.state.extra.items():
        assert np.asarray(back.state.extra[key]).tobytes() == np.asarray(value).tobytes(), key
    bad = dict(carry.state.extra, tgr=jnp.zeros((5,), jnp.float32))
    with pytest.raises(ValueError, match='E78'):
        validate_urban_bep(carry._replace(state=carry.state._replace(extra=bad)), {'sf_urban_physics': 3}, grid_shape=(4, 5), nz=8)


def _extra_mutants(carry, n):
    ex = carry.state.extra
    return {
        'missing_ep_pv': {k: v for k, v in ex.items() if k != 'ep_pv'},
        'missing_gr_temperature': {k: v for k, v in ex.items() if k != 'gr_temperature'},
        'ep_pv_wrong_rank': dict(ex, ep_pv=jnp.zeros((n, 2), jnp.float32)),
        'gr_temperature_wrong_axes': dict(ex, gr_temperature=jnp.zeros((n, 1, 1, 10), jnp.float32)),
        'tgr_wrong_rank': dict(ex, tgr=jnp.zeros((n, 2), jnp.float32)),
        'all_missing': {},
        'unknown_key': dict(ex, unknown=jnp.zeros((n,), jnp.float32)),
    }


MUTANTS = ('missing_ep_pv', 'missing_gr_temperature', 'ep_pv_wrong_rank', 'gr_temperature_wrong_axes',
           'tgr_wrong_rank', 'all_missing', 'unknown_key')


@pytest.mark.parametrize('kind', MUTANTS)
def test_public_restart_validator_refuses_incomplete_pv_green_state(tmp_path, kind):
    """rv F1: the active PV/green-roof key set and every shape are bound (live bundle or saved selection)."""
    from gpuwrf.io.restart_schema import validate_urban_bep
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    live = {'sf_urban_physics': 3, 'urban_bundle': bundle}
    assert validate_urban_bep(carry, live, grid_shape=(4, 5), nz=8) == ('green_roof', 'pv')
    bad = carry._replace(state=carry.state._replace(extra=_extra_mutants(carry, len(bundle.indices))[kind]))
    for namelist, bound in ((live, None), ({'sf_urban_physics': 3}, ['green_roof', 'pv'])):
        with pytest.raises(ValueError, match='E78'):
            validate_urban_bep(bad, namelist, grid_shape=(4, 5), nz=8, extensions=bound)


def test_restart_selection_binding(tmp_path):
    from gpuwrf.io.restart_schema import validate_urban_bep
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    live = {'sf_urban_physics': 3, 'urban_bundle': bundle}
    for bound in (['pv'], ['green_roof'], [], ['pv', 'green_roof'], ['green_roof', 'pv', 'pv'], ['solar'], 'pv'):
        with pytest.raises(ValueError, match='E78'):
            validate_urban_bep(carry, live, grid_shape=(4, 5), nz=8, extensions=bound)
    # Unbound legacy reads accept only a complete selection's key set.
    assert validate_urban_bep(carry, {'sf_urban_physics': 3}, grid_shape=(4, 5), nz=8) == ('green_roof', 'pv')
    hook, off_bundle, off = _hook_case(tmp_path, 'ctl_ac_on')
    off_live = {'sf_urban_physics': 3, 'urban_bundle': off_bundle}
    assert validate_urban_bep(off, off_live, grid_shape=(4, 5), nz=8) == ()
    with pytest.raises(ValueError, match='E78'):  # PV/green-roof state under a PV/GR-off selection
        validate_urban_bep(off._replace(state=off.state._replace(extra=carry.state.extra)), off_live, grid_shape=(4, 5), nz=8)


def test_native_snapshot_binds_active_pv_green_selection(tmp_path):
    from gpuwrf.runtime.restart_store import _urban_bep_restart_config, _validate_urban_bep_snapshot
    hook, bundle, carry = _hook_case(tmp_path, 'gr10pv03_ac_on')
    hook, off_bundle, off = _hook_case(tmp_path, 'ctl_ac_on')
    grid = SimpleNamespace(nz=8, ny=4, nx=5)
    config = _urban_bep_restart_config(SimpleNamespace(grid=grid, sf_urban_physics=3, urban_bundle=bundle))
    assert config == {'sf_urban_physics': 3, 'domain_shape': [8, 4, 5], 'bem_extensions': ['green_roof', 'pv']}
    off_config = _urban_bep_restart_config(SimpleNamespace(grid=grid, sf_urban_physics=3, urban_bundle=off_bundle))
    assert off_config == {'sf_urban_physics': 3, 'domain_shape': [8, 4, 5]}  # PV/GR-off identity unchanged

    def check(group, cfg):
        payload = {'state_fields': {'theta': np.zeros((8, 4, 5), np.float32)}, 'extra_fields': {'urban_bep': group}}
        _validate_urban_bep_snapshot(payload, {'d01': cfg}, ['d01'], 'd01', True)
    check(carry, config)
    check(off, off_config)
    for kind, extra in _extra_mutants(carry, len(bundle.indices)).items():
        with pytest.raises(ValueError, match='E78'):
            check(carry._replace(state=carry.state._replace(extra=extra)), config)
    for cfg in (off_config, dict(config, bem_extensions=['pv']), dict(config, extra_key=1)):
        with pytest.raises(ValueError, match='E78'):
            check(carry, cfg)


@pytest.fixture
def pv_green_case(tmp_path, monkeypatch):
    """test_v040_urban_boundaries.urban_case with the gr10pv03 URBPARM.TBL (real carry/namelist/writers)."""
    import shutil
    from gpuwrf.config import paths
    from run_oracle import ARMS, table
    from test_v040_urban_boundaries import urban_case
    _require_pristine_tables()
    run_dir = tmp_path/'run';run_dir.mkdir()
    for name in ('URBPARM.TBL', 'MPTABLE.TBL'):
        shutil.copy(paths.wrf_run_dir()/name, run_dir/name)
    (run_dir/'URBPARM.TBL').write_text(table(ARMS['gr10pv03_ac_on']))
    monkeypatch.setattr(paths, 'wrf_run_dir', lambda: run_dir)
    return urban_case.__wrapped__(monkeypatch)(3)


@pytest.mark.parametrize('fmt', ['pickle', 'netcdf'])
def test_pickle_and_wrfrst_checkpoints_bind_pv_green_state(tmp_path, monkeypatch, pv_green_case, fmt):
    from gpuwrf.io import restart_schema as schema
    from test_restart_schema_held_physics import exact, read, save
    grid, nl, carry = pv_green_case
    assert set(carry.urban_bep.state.extra) == {'ep_pv', *__import__('gpuwrf.physics.urban_bem_model',
        fromlist=['GR_EXTRA']).GR_EXTRA}
    good = tmp_path/'healthy';bad_path = tmp_path/'malformed'
    save(good, fmt, grid, nl, carry)
    if fmt == 'netcdf':
        import netCDF4
        with netCDF4.Dataset(good) as ds:
            assert 'GPUWRF_URBAN_BEM_EXTENSIONS' in ds.ncattrs()
            assert json.loads(ds.getncattr('GPUWRF_URBAN_BEM_EXTENSIONS')) == ['green_roof', 'pv']
    exact(carry, read(good, fmt))
    for kind, extra in _extra_mutants(carry.urban_bep, len(nl.urban_bundle.indices)).items():
        bad = carry.replace(urban_bep=carry.urban_bep._replace(state=carry.urban_bep.state._replace(extra=extra)))
        with pytest.raises(ValueError, match='E78'):
            save(tmp_path/f'refused_{kind}', fmt, grid, nl, bad)
        # A PV/GR run's file corrupted after write: bypass only write-time admission (selection still bound).
        with monkeypatch.context() as bypass:
            bypass.setattr(schema, 'validate_urban_bep', lambda *a, **k: ('green_roof', 'pv'))
            save(bad_path, fmt, grid, nl, bad)
        with pytest.raises(ValueError, match='E78'):
            read(bad_path, fmt)
        bad_path.unlink()
