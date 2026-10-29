"""Pristine-WRF output transcription, selected native outputs, and E78 gates."""
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.ice_diagnostics import ICE_DIAGNOSTIC_LEAVES, ice_diagnostic_leaves

ROOT = Path(__file__).resolve().parents[1]


def helper(file):
    spec = importlib.util.spec_from_file_location('ice_' + file, ROOT / 'tests' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_outputs(mp, step=1):
    """Independent wrapper output names; never construct truth from product aliases."""
    if mp == 55:
        original = json.loads((ROOT / 'proofs/s2_mpmore/jensen55/fixtures/fp32_10.json').read_text())
        out = {n: np.asarray(a, np.float32)[None, :] for n, a in original['output'].items()}
        expected = {}
        for cat in (1, 2, 3):
            suffix = str(cat) if cat > 1 else ''
            for source, target in (('di', 'd_ice'), ('rho', 'rho_ice'), ('vm', 'v_ice'), ('phi', 'phi_ice')):
                expected[target + suffix] = out[source + str(cat)]
        return out, expected
    with np.load(ROOT / f'proofs/s2_p3/fixtures/columns_mp{mp}.npz') as data:
        out = {n: data[n][..., step].copy() for n in data.files if data[n].ndim == 3}
    expected = {}
    for suffix in (('', '2') if mp == 52 else ('',)):
        for source, target in (('di', 'd_ice'), ('rhoi', 'rho_ice'), ('vm', 'v_ice')):
            expected[target + suffix] = out[source + suffix]
    return out, expected


@pytest.mark.parametrize('mp,count', [(8, 0), (50, 3), (51, 3), (52, 6), (53, 3), (55, 12)])
def test_selected_allocation_and_real_lock(mp, count, monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])
    h = helper('test_rrtm_lw_operational_wiring.py'); grid = h._grid()
    state = sc.State.zeros(grid, mp_physics=mp).ensure_conditional_leaves(mp_physics=mp)
    active = {n for n in ICE_DIAGNOSTIC_LEAVES if getattr(state, n) is not None}
    assert active == set(ice_diagnostic_leaves(mp)) and len(active) == count
    from gpuwrf.contracts.precision import MYNN_CLOUDPDF_LEAVES
    assert sc.State.__slots__[-15:-3] == ICE_DIAGNOSTIC_LEAVES
    assert sc.State.__slots__[-3:] == MYNN_CLOUDPDF_LEAVES
    forced = _enforce_operational_precision(state, force_fp64=True)
    for n in active:
        assert DEFAULT_DTYPES.dtype_for(n) == jnp.float32
        assert getattr(forced, n).dtype == jnp.float32
        np.testing.assert_array_equal(getattr(state, n), 0.)  # WRF Registry initial value.


@pytest.mark.parametrize('mp', [50, 51, 52, 53, 55])
@pytest.mark.parametrize('native', [False, True])
def test_adapter_retains_original_outputs_once_at_call_time(mp, native, monkeypatch):
    from gpuwrf.backends import pallas as backend
    from gpuwrf.coupling.physics_couplers import _from_columns
    from gpuwrf.physics.p3 import adapter as p3
    from gpuwrf.physics.jensen_ishmael import adapter as jensen, column
    out, expected = original_outputs(mp)
    ncol, nz = next(iter(expected.values())).shape
    h = helper('test_rrtm_lw_operational_wiring.py'); grid = h._grid(nz=nz, ny=1, nx=ncol)
    state = h._state(grid).ensure_conditional_leaves(mp_physics=mp)
    state = state.replace(**{n: jnp.full_like(state.qv, -17., dtype=jnp.float32) for n in expected})
    monkeypatch.setattr(backend, 'cuda_native', lambda: native)
    calls = []

    def emit(fields, dt, timestep, **kwargs):
        calls.append((fields, timestep, kwargs))
        assert all(n not in fields for n in expected)  # No diagnostic feedback into the producer.
        if native:
            needed = set(expected)
            selected = set(kwargs['output_names'])
            independent = {('rhoi' if n.startswith('rho_') and mp != 55 else
                            'rho' if n.startswith('rho_') else 'di' if n.startswith('d_') else
                            'vm' if n.startswith('v_') else 'phi') +
                           (n[-1] if n[-1].isdigit() else '1' if mp == 55 else '') for n in needed}
            assert independent <= selected
        result = {n: jnp.asarray(a) for n, a in out.items()}
        z = jnp.zeros((ncol,), jnp.float32)
        return (result, dict(rainstep=z, snowstep=z)) if mp == 55 else dict(result, rainncv=z, snowncv=z)

    if native:
        from gpuwrf.kernels import phys_p3, phys_jensen
        monkeypatch.setattr(phys_jensen if mp == 55 else phys_p3, 'full_column', emit)
    else:
        monkeypatch.setattr(column if mp == 55 else p3, 'step', emit)
    adapter = jensen.jensen_adapter if mp == 55 else lambda *a, **kw: p3.p3_adapter(*a, mp=mp, **kw)
    updated = adapter(state, 18., grid, timestep=7)
    assert len(calls) == 1 and calls[0][1] == 7
    for n, truth in expected.items():
        actual = np.asarray(getattr(updated, n))
        wanted = np.asarray(_from_columns(jnp.asarray(truth)))
        assert np.isfinite(actual).all() and actual.dtype == np.float32
        assert actual.tobytes() == wanted.tobytes(), n
    assert any(np.any(a > 0) for a in expected.values())  # Active original ice, no dry-only happy path.


@pytest.mark.parametrize('mp', [50, 51, 52, 53, 55])
def test_real_checkpoint_and_netcdf_restart_preserve_and_refuse_old_schema(mp, tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint, read_checkpoint_with_runtime_state
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry, _validate_state_field_order
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.restart_schema import validate_held_state
    h = helper('test_rrtm_lw_operational_wiring.py'); grid = h._grid()
    nl = replace(h._namelist(grid), mp_physics=mp)
    state = h._state(grid).ensure_conditional_leaves(mp_physics=mp)
    state = state.replace(**{n: jnp.full_like(state.qv, (i + 1) / 32, dtype=jnp.float32)
                             for i, n in enumerate(ice_diagnostic_leaves(mp))})
    checkpoint = tmp_path / 'restart.pkl'
    carry=initial_operational_carry(state)
    # P3 and Jensen hold REFL_10CM from every MP call (always-radar schemes, E78).
    carry=carry.replace(radar_reflectivity=jnp.full_like(state.qv,-7,jnp.float32))
    write_checkpoint(state, nl, grid, 7, checkpoint, runtime_state=carry)
    restored, _, _, step, restored_carry = read_checkpoint_with_runtime_state(checkpoint); assert step == 7
    restart = tmp_path / 'restart.nc'
    write_wrfrst_carry(carry, grid, nl, restart, valid_time='2026-07-25_18:02:06',
                       run_start='2026-07-25_18:00:00', step_index=7)
    restored_nc_carry, _ = read_wrfrst_carry(restart)
    restored_nc=restored_nc_carry.state
    assert np.asarray(restored_carry.radar_reflectivity).tobytes()==np.asarray(carry.radar_reflectivity).tobytes()
    assert np.asarray(restored_nc_carry.radar_reflectivity).tobytes()==np.asarray(carry.radar_reflectivity).tobytes()
    fields = {n: getattr(state, n) for n in state.__slots__}
    for n in ice_diagnostic_leaves(mp):
        for got in (restored, restored_nc):
            assert np.asarray(getattr(got, n)).tobytes() == np.asarray(fields[n]).tobytes()
        for bad in (None, np.full(state.qv.shape, np.nan, np.float32),
                    np.full(state.qv.shape, np.inf, np.float32),
                    np.full(state.qv.shape, -np.inf, np.float32),
                    np.zeros(state.qv.shape, np.float64), np.zeros((1,), np.float32)):
            with pytest.raises(ValueError, match='held ice diagnostic.*E78'):
                validate_held_state({**fields, n: bad}, nl)
    with pytest.raises(ValueError, match='held ice diagnostics.*E78'):
        _validate_state_field_order(tuple(n for n in state.__slots__ if n not in ICE_DIAGNOSTIC_LEAVES), namelist=nl)
    # The public reader must refuse old payloads; raw State unpickle compatibility is insufficient.
    payload = pickle.loads(checkpoint.read_bytes())
    removed = ice_diagnostic_leaves(mp)[0]
    payload['state_fields'].pop(removed)
    payload['state_field_order'] = tuple(n for n in payload['state_field_order'] if n != removed)
    checkpoint.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match='held ice diagnostic.*E78'):
        read_checkpoint(checkpoint)


@pytest.mark.parametrize('mp', [50, 52, 55])
@pytest.mark.parametrize('mode', ['full', 'operational', 'subset'])
def test_writer_copies_original_diagnostics_and_metadata(mp, mode, tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    h = helper('test_m7_netcdf_writer.py')
    _, expected = original_outputs(mp); ncol, nz = next(iter(expected.values())).shape
    state, grid, nl = h.synthetic_case(nx=ncol, ny=1, nz=nz); nl['mp_physics'] = mp
    for leaf, value in expected.items():
        setattr(state, leaf, value.T[:, None, :])
    names = tuple(n.upper() for n in expected)
    subset = names if mode == 'subset' else None
    start = datetime(2023, 1, 15, tzinfo=timezone.utc); authority = h.writer_authority(grid, 'd01')
    payload = prepare_wrfout_payload(state, grid, nl, tmp_path / 'history.nc', domain='d01',
        domain_authority=authority, valid_time=start, run_start=start, lead_hours=0,
        full_variable_set=mode != 'operational', variable_subset=subset)
    write_prepared_wrfout(payload, expected_domain='d01', expected_domain_authority=authority, variable_subset=subset)
    with Dataset(payload.target) as ds:
        for leaf, value in expected.items():
            var = ds[leaf.upper()]
            assert var.dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')
            assert var.dtype == np.float32 and var.MemoryOrder == 'XYZ' and var.FieldType == 104
            assert np.asarray(var[0]).tobytes() == value.T[:, None, :].tobytes()
            assert var.units == ('kg m-3' if leaf.startswith('rho') else 'm s-1' if leaf.startswith('v')
                                 else '' if leaf.startswith('phi') else 'm')
        if mp != 55:
            assert not any(n.startswith('PHI_ICE') for n in ds.variables)


@pytest.mark.parametrize('leaf', ICE_DIAGNOSTIC_LEAVES)
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_writer_refused(leaf, bad, tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    h = helper('test_m7_netcdf_writer.py'); state, grid, nl = h.synthetic_case(); nl['mp_physics'] = 55
    a = np.zeros(state.qv.shape, np.float32); a.flat[0] = bad; setattr(state, leaf, a)
    start = datetime(2023, 1, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='nonfinite microphysics history field ' + leaf.upper()):
        prepare_wrfout_payload(state, grid, nl, tmp_path / 'bad.nc', domain='d01',
            domain_authority=h.writer_authority(grid, 'd01'), valid_time=start, run_start=start, lead_hours=0,
            full_variable_set=True, variable_subset=(leaf.upper(),))


@pytest.mark.parametrize('mp', [50, 52, 55])
@pytest.mark.parametrize('damage', ['missing', 'nan', 'dtype'])
def test_public_native_restart_rehashed_corruption_refused(mp, damage, tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore, nested_identity
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.io.restart import write_restart, read_restart
    h = helper('test_rrtm_lw_operational_wiring.py'); grid = h._grid()
    nl = replace(h._namelist(grid), mp_physics=mp)
    state = h._state(grid).ensure_conditional_leaves(mp_physics=mp)
    state = state.replace(**{n: jnp.full_like(state.qv, (i+1)/32, dtype=jnp.float32)
                             for i, n in enumerate(ice_diagnostic_leaves(mp))})
    carry = initial_operational_carry(state)
    # P3 and Jensen hold REFL_10CM from every MP call (always-radar schemes, E78).
    carry=carry.replace(radar_reflectivity=jnp.full_like(state.qv,-7,jnp.float32))
    path = tmp_path / 'carry.pkl'; write_restart(carry, nl, grid, 7, path)
    restored = read_restart(path)[0]
    for n in ice_diagnostic_leaves(mp):
        assert np.asarray(getattr(restored.state, n)).tobytes() == np.asarray(getattr(state, n)).tobytes()
    config = NestedPipelineConfig(tmp_path, tmp_path/'out', tmp_path/'proof', hours=1, max_dom=1)
    identity = nested_identity(config, {'d01': SimpleNamespace(namelist=nl)}, datetime(2023,1,15,tzinfo=timezone.utc))
    store = RestartStore(tmp_path/'store', 10**8, 2, 0)
    generation, _ = store.save({'d01': carry}, {'d01': 7}, {'d01': nl.dt_s}, identity)
    native = store.read(generation, expected_identity=identity)[0]['carries']['d01']
    for n in ice_diagnostic_leaves(mp):
        assert np.asarray(getattr(native.state, n)).tobytes() == np.asarray(getattr(state, n)).tobytes()
    payload = pickle.loads((generation/'snapshot.pkl').read_bytes())
    saved = payload['carries']['d01']; leaf = ice_diagnostic_leaves(mp)[0]
    if damage == 'missing':
        saved['state_fields'].pop(leaf)
        saved['state_field_order'] = [n for n in saved['state_field_order'] if n != leaf]
    elif damage == 'nan':
        saved['state_fields'][leaf] = saved['state_fields'][leaf].copy()
        saved['state_fields'][leaf].flat[0] = np.nan
    else:
        saved['state_fields'][leaf] = saved['state_fields'][leaf].astype(np.float64)
    data = pickle.dumps(payload); (generation/'snapshot.pkl').write_bytes(data)
    manifest = json.loads((generation/'manifest.json').read_text())
    manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (generation/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='held ice diagnostic.*E78'):
        store.read(generation)
