"""BEP+BEM (sf_urban_physics=3) green-roof/PV history: QGR_URB4D, TGR_URB4D, DRAINGR_URB3D, EP_PV_URB3D.

Reference: original CPU-WRF composite c57 (Madrid BEP+BEM, Noah-MP), every cell of all 25 frames
(proofs/t_classd/URBAN_CPU_ALL_FRAMES.json) and its 42x42 extract (c57_green_roof_extract.npz):
QGR(t0) == Noah-MP SMOIS top layer; QGR(t>0) keeps it except urban columns, which bep_bem resets
to 0; the three other fields are +0 with WRF's green roof / PV branches off (the only admitted case).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import urban_multilayer_hook as hook

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / 'proofs/t_classd'
NAMES = ('QGR_URB4D', 'TGR_URB4D', 'DRAINGR_URB3D', 'EP_PV_URB3D')


def _extract():
    return np.load(PROOF / 'c57_green_roof_extract.npz')


def _bundle(ivgtyp, isurban, option=3):
    from gpuwrf.config.paths import wrf_run_dir
    return hook.load_multilayer_bundle(wrf_run_dir(), option=option, ivgtyp=ivgtyp, isurban=isurban)


def _seed(x, option=3):
    bundle = _bundle(x['ivgtyp'], int(x['isurban']), option)
    ny, nx = x['ivgtyp'].shape
    smois = np.stack([x['smois_top_t0']] + [np.full((ny, nx), .25, np.float32)] * 3)
    carry = hook.initial_multilayer_carry(bundle, tslb=jnp.full((4, ny, nx), 290, jnp.float32),
                                          frc_urb2d=jnp.zeros((ny, nx), jnp.float32), nz=8, smois=smois)
    return bundle, carry


def test_original_cpu_proof_is_complete():
    proof = json.loads((PROOF / 'URBAN_CPU_ALL_FRAMES.json').read_text())
    assert proof['summary']['all_match'] and len(proof['frames']) == 25
    assert proof['summary']['urban_columns'] == 31 and proof['summary']['qgr_t0_nonzero'] == 42 * 42
    for frame in proof['frames']:
        for name in NAMES:
            assert frame['fields'][name]['match'], (frame['file'], name)
    x = _extract()
    assert np.count_nonzero(x['qgr_t0']) == x['qgr_t0'].size
    assert np.count_nonzero(x['qgr_last'] == 0) == np.count_nonzero(x['ivgtyp'] == x['isurban']) == 31


def test_urban_var_init_seeds_qgr_from_smois_top_like_original_cpu():
    x = _extract()
    _bundle_, carry = _seed(x)
    assert carry.green_roof.dtype == jnp.float32 and carry.green_roof.shape == x['qgr_t0'].shape
    assert np.asarray(carry.green_roof).tobytes() == x['qgr_t0'].tobytes()


def test_bep_only_option_has_no_green_roof_and_bem_requires_smois():
    x = _extract()
    assert _seed(x, option=2)[1].green_roof is None
    bundle = _bundle(x['ivgtyp'], int(x['isurban']), 3)
    with pytest.raises(ValueError, match='SMOIS'):
        hook.initial_multilayer_carry(bundle, tslb=jnp.full((4, 42, 42), 290, jnp.float32),
                                      frc_urb2d=jnp.zeros((42, 42)), nz=8)


def test_bep_call_resets_urban_columns_like_original_cpu(monkeypatch):
    """apply_urban_multilayer carry handling with stubbed column physics (transcription only)."""
    x = _extract()
    bundle, carry = _seed(x)
    n = len(bundle.indices)
    import gpuwrf.physics.urban_bem_model as bem
    monkeypatch.setattr(bem, 'bem_column_step', lambda state, *a, **k: SimpleNamespace(
        state=state, moisture_source=jnp.zeros((n,), jnp.float32)))
    surface = SimpleNamespace(**{k: jnp.ones((n,), jnp.float32) for k in
                                 ('skin', 'heat', 'moisture', 'latent', 'ground', 'albedo', 'emissivity', 'ust')})
    monkeypatch.setattr(hook, 'noahmp_bep_blend', lambda *a, **k: (surface, carry.sources))
    grid = jnp.zeros((42, 42), jnp.float32)
    nm = SimpleNamespace(tsk=grid, hfx=grid, qfx=grid, grdflx=grid, albedo=grid, emiss=grid, lh=grid, history=None,
                         _replace=lambda **k: None)
    profile = SimpleNamespace(theta=jnp.zeros((n, 8)), u=jnp.zeros((n, 8)), v=jnp.zeros((n, 8)))
    forcing = SimpleNamespace(prcpconv=grid, prcpnonc=grid)
    out = jax.jit(lambda c: hook.apply_urban_multilayer(
        bundle, c, atmosphere=(profile, jnp.zeros((n, 8)), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n)),
        forcing=forcing, nm=nm, ust=grid, dt=50.)[2])(carry)
    assert np.asarray(out.green_roof).tobytes() == x['qgr_last'].tobytes()
    again = hook.apply_urban_multilayer(
        bundle, out, atmosphere=(profile, jnp.zeros((n, 8)), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n)),
        forcing=forcing, nm=nm, ust=grid, dt=50.)[2]
    assert np.asarray(again.green_roof).tobytes() == x['qgr_last'].tobytes()  # idempotent like WRF


def test_restart_requires_green_roof_for_bem_only():
    from gpuwrf.io.restart_schema import validate_urban_bep
    x = _extract()
    _b, carry = _seed(x)
    validate_urban_bep(carry, {'sf_urban_physics': 3}, grid_shape=(42, 42), nz=8)
    with pytest.raises(ValueError, match='QGR_URB3D'):
        validate_urban_bep(carry._replace(green_roof=None), {'sf_urban_physics': 3}, grid_shape=(42, 42), nz=8)
    with pytest.raises(ValueError, match='QGR_URB3D grid shape'):
        validate_urban_bep(carry, {'sf_urban_physics': 3}, grid_shape=(41, 42), nz=8)
    with pytest.raises(ValueError, match='(E78)'):
        validate_urban_bep(carry._replace(green_roof=carry.green_roof.astype(jnp.float64)),
                           {'sf_urban_physics': 3}, grid_shape=(42, 42), nz=8)
    _b2, bep = _seed(x, option=2)
    validate_urban_bep(bep, {'sf_urban_physics': 2}, grid_shape=(42, 42), nz=8)
    with pytest.raises(ValueError, match='QGR_URB3D'):
        validate_urban_bep(bep._replace(green_roof=carry.green_roof), {'sf_urban_physics': 2},
                           grid_shape=(42, 42), nz=8)


@pytest.mark.parametrize('full', (False, True))
def test_writer_emits_original_schema_and_values(tmp_path, full):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    proof = json.loads((PROOF / 'URBAN_CPU_ALL_FRAMES.json').read_text())['frames'][1]['fields']
    state, grid, nl = synthetic_case()
    nl.update(sf_urban_physics=3)
    qgr = np.arange(20, dtype=np.float32).reshape(4, 5) / np.float32(40)
    auth = writer_authority(grid, 'd01')
    p = prepare_wrfout_payload(state, grid, nl, tmp_path / 'urban', domain='d01', domain_authority=auth,
                               valid_time='2026-10-13T01:00:00', run_start='2026-10-13T00:00:00', lead_hours=1.,
                               full_variable_set=full, variable_subset=NAMES, diagnostics={'QGR_URB4D': qgr})
    write_prepared_wrfout(p, expected_domain='d01', expected_domain_authority=auth, variable_subset=NAMES)
    with Dataset(p.target) as ds:
        for name in NAMES:
            v = ds[name]
            ref = proof[name]
            assert list(v.dimensions) == ref['dims'] and str(v.dtype) == ref['dtype']
            assert v.units == ref['units'] and v.stagger == ref['stagger']
            data = np.asarray(v[0], np.float32)
            if name == 'QGR_URB4D':
                assert data.tobytes() == qgr.tobytes()
            else:
                assert np.all(data.view(np.uint32) == 0)


def test_writer_refuses_nonzero_supplied_inactive_urban_field(tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    state, grid, nl = synthetic_case()
    nl.update(sf_urban_physics=3)
    with pytest.raises(ValueError, match='EP_PV_URB3D'):
        prepare_wrfout_payload(state, grid, nl, tmp_path / 'bad', domain='d01',
                               domain_authority=writer_authority(grid, 'd01'),
                               valid_time='2026-10-13T01:00:00', run_start='2026-10-13T00:00:00', lead_hours=1.,
                               variable_subset=NAMES, diagnostics={'EP_PV_URB3D': np.ones((4, 5), np.float32)})


def test_non_bem_runs_do_not_receive_urban_fields():
    from gpuwrf.io.producer_history import inactive_names
    lookup = lambda obj, key, default=None: obj.get(key, default) if isinstance(obj, dict) else default
    for option in (0, 1, 2):
        assert not set(inactive_names({'sf_urban_physics': option}, lookup)) & set(NAMES)
    assert set(inactive_names({'sf_urban_physics': 3}, lookup)) >= set(NAMES[1:])


@pytest.fixture
def _urban_native(monkeypatch, tmp_path):
    from test_v040_urban_boundaries import urban_case
    from test_online_diagnostics_restart_store import identity, rewrite
    from gpuwrf.runtime.restart_store import RestartStore
    build = urban_case.__wrapped__(monkeypatch)
    return build, identity, rewrite, lambda tag: RestartStore(tmp_path / tag, 10**7, 2, 0)


def _corrupt(urban, kind):
    leaf = urban.green_roof
    if kind == 'missing':
        return urban._replace(green_roof=None)
    if kind == 'shape':
        return urban._replace(green_roof=np.ones((1, 1), np.float32))
    if kind == 'precision':
        return urban._replace(green_roof=np.asarray(leaf, np.float64))
    if kind == 'nan':
        bad = np.array(leaf, np.float32); bad.flat[0] = np.nan
        return urban._replace(green_roof=bad)
    return None  # whole BEP group deleted


@pytest.mark.parametrize('kind', ['missing', 'shape', 'precision', 'nan', 'group'])
def test_native_checkpoint_refuses_bad_qgr_on_save_and_authenticated_read(_urban_native, kind):
    """rv-t-classd F1: the public native store binds BEP/BEM selection+domain and validates QGR_URB3D."""
    build, identity, rewrite, store_for = _urban_native
    grid, nl, carry = build(3)
    bad = carry.replace(urban_bep=_corrupt(carry.urban_bep, kind))
    with pytest.raises(ValueError, match='E78'):
        store_for('save').save({'d01': bad}, {'d01': 2}, {'d01': nl.dt_s}, identity(nl))
    store = store_for('read')
    generation, _ = store.save({'d01': carry}, {'d01': 2}, {'d01': nl.dt_s}, identity(nl))

    def change(payload, manifest):
        payload['extra_fields']['urban_bep'] = _corrupt(payload['extra_fields']['urban_bep'], kind)
    rewrite(generation, change)  # recomputed hash/size: only the schema can refuse
    with pytest.raises(ValueError, match='E78|non-finite|nonfinite'):
        store.read(generation, expected_identity=identity(nl))


def test_native_checkpoint_roundtrips_qgr_bytes_and_binding(_urban_native):
    build, identity, rewrite, store_for = _urban_native
    grid, nl, carry = build(3)
    ident = identity(nl)
    assert ident['urban_bep_config']['d01'] == {'sf_urban_physics': 3, 'domain_shape': [grid.nz, grid.ny, grid.nx]}
    store = store_for('ok')
    generation, _ = store.save({'d01': carry}, {'d01': 2}, {'d01': nl.dt_s}, ident)
    back = store.read(generation, expected_identity=ident)[0]['carries']['d01']
    assert np.asarray(back.urban_bep.green_roof).tobytes() == np.asarray(carry.urban_bep.green_roof).tobytes()
    unbound = {k: v for k, v in ident.items() if k != 'urban_bep_config'}
    with pytest.raises(ValueError, match='E78'):
        store_for('unbound').save({'d01': carry}, {'d01': 2}, {'d01': nl.dt_s}, unbound)
