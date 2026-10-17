"""Optional surface history admission follows its original WRF producer."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.restart_schema import history_field_required, validate_history_winds
from gpuwrf.runtime.operational_state import HistoryDiagnostics


@pytest.mark.parametrize('scheme', [1, 2, 4, 5, 7, 10])
@pytest.mark.parametrize('representation', ['object', 'flat', 'physics'])
def test_history_field_activity(scheme, representation):
    options = {'sf_sfclay_physics': scheme}
    nml = (SimpleNamespace(**options) if representation == 'object' else
           {'physics': options} if representation == 'physics' else options)
    # WRF surface_driver MYJSFC/QNSESFC supply the equivalent-roughness pair.
    for field in ('u10e', 'v10e'):
        assert history_field_required(field, nml) == (scheme in (2, 4))
    assert all(history_field_required(field, nml) for field in ('t2','q2','u10','v10','lh','psfc'))
    assert not history_field_required('hfx', nml)
    z = np.zeros((2, 3), np.float32)
    history = HistoryDiagnostics(*(z for _ in range(6)))
    if scheme in (2, 4):
        with pytest.raises(ValueError, match=r'history.u10e'):
            validate_history_winds(history, nml, shape=(2, 3))
    else:
        validate_history_winds(history, nml, shape=(2, 3))


@pytest.mark.parametrize('scheme', [2, 4])
@pytest.mark.parametrize('field', ['u10e', 'v10e'])
@pytest.mark.parametrize('damage', ['missing', 'shape', 'dtype', 'nan', 'inf', 'minus_inf'])
def test_active_history_wind_damage_refused(scheme, field, damage):
    z = np.zeros((2, 3), np.float32)
    history = HistoryDiagnostics(*(z for _ in range(6)), u10e=z+4.5, v10e=z-2.5)
    bad = (None if damage == 'missing' else np.zeros((1, 1), np.float32) if damage == 'shape'
           else z.astype(np.float64) if damage == 'dtype' else
           np.full_like(z, {'nan': np.nan, 'inf': np.inf, 'minus_inf': -np.inf}[damage]))
    with pytest.raises(ValueError, match='history.'+field):
        validate_history_winds(history._replace(**{field: bad}), {'sf_sfclay_physics': scheme}, shape=(2, 3))


@pytest.mark.parametrize('surface', [1, 4])
def test_qnse_pbl_uses_resolved_surface_history_activity(surface):
    from gpuwrf.coupling.qnse_adapter import QnseCarry
    from gpuwrf.io.restart_schema import validate_qnse
    z = np.zeros((2, 3), np.float32)
    profile = np.zeros((6, 2, 3), np.float32)
    group = QnseCarry(*(z for _ in range(9)), profile, profile, profile)
    history = HistoryDiagnostics(*(z for _ in range(6)),
                                 u10e=z if surface == 4 else None,
                                 v10e=z if surface == 4 else None)
    validate_qnse(group, {'bl_pbl_physics': 4, 'sf_sfclay_physics': surface},
                  1, history=history, expected_shape=profile.shape)
    with pytest.raises(ValueError, match='QNSE output t2'):
        validate_qnse(group, {'bl_pbl_physics': 4, 'sf_sfclay_physics': surface},
                      1, history=history._replace(t2=None), expected_shape=profile.shape)


def _myj():
    from test_v013_myj_janjic_operational import _grid, _state, _namelist
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    grid = _grid()
    state = _state(grid)
    history = seed_history_diagnostics(state, myjsfc=True)
    history = history._replace(u10e=history.u10e+4.5, v10e=history.v10e-2.5)
    from gpuwrf.coupling.myj_surface import initial_myj_memory
    held = initial_myj_memory(state)
    # Distinct nonzero MYJSFC memory (CALLER_AUDIT F12) detects cold starts or leaf swaps.
    held = type(held)(*(jnp.arange(a.size, dtype=jnp.float32).reshape(a.shape) / 512 + i
                        for i, a in enumerate(held, 1)))
    carry = initial_operational_carry(state).replace(history_diagnostics=history, qnse=held)
    nml = _namelist(grid, bl_pbl_physics=2, sf_sfclay_physics=2)
    identity = {'temf_config': {'d01': {'bl_pbl_physics': 2, 'sf_sfclay_physics': 2,
                                      'domain_shape': [grid.nz, grid.ny, grid.nx]}}}
    return grid, nml, carry, identity


def _write_read(tmp_path, format, grid, nml, carry, identity):
    if format == 'pickle':
        from gpuwrf.io.restart import write_restart, read_restart
        path = tmp_path/'carry.pkl'
        write_restart(carry, nml, grid, 7, path)
        return read_restart(path)[0]
    if format == 'netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
        path = tmp_path/'carry.nc'
        write_wrfrst_carry(carry, grid, nml, path, valid_time='2023-01-15_00:02:06',
                          run_start='2023-01-15_00:00:00', step_index=7)
        return read_wrfrst_carry(path, namelist=nml)[0]
    from gpuwrf.runtime.restart_store import RestartStore
    store = RestartStore(tmp_path/'native', 10_000_000, 2, 0)
    path, _ = store.save({'d01': carry}, {'d01': 7}, {'d01': 18.}, identity)
    return store.read(path, expected_identity=identity)[0]['carries']['d01']


@pytest.mark.parametrize('format', ['pickle', 'netcdf', 'native'])
def test_myj_public_formats_retain_complete_carry(cpu_pallas_interpret, tmp_path, format):
    grid, nml, carry, identity = _myj()
    restored = _write_read(tmp_path, format, grid, nml, carry, identity)
    assert jax.tree.structure(restored) == jax.tree.structure(carry)
    for a, b in zip(jax.tree.leaves(carry), jax.tree.leaves(restored), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@pytest.mark.parametrize('format', ['pickle', 'netcdf', 'native'])
@pytest.mark.parametrize('field', ['u10e', 'v10e'])
def test_myj_public_formats_refuse_missing_wind(cpu_pallas_interpret, tmp_path, format, field):
    grid, nml, carry, identity = _myj()
    carry = carry.replace(history_diagnostics=carry.history_diagnostics._replace(**{field: None}))
    with pytest.raises(ValueError, match='history.'+field):
        _write_read(tmp_path, format, grid, nml, carry, identity)


@pytest.mark.parametrize('field', ['u10e', 'v10e'])
@pytest.mark.parametrize('format', ['pickle', 'native', 'netcdf'])
@pytest.mark.parametrize('damage', ['missing', 'dtype', 'nonfinite'])
def test_myj_public_reads_refuse_damaged_wind(cpu_pallas_interpret, tmp_path, format, field, damage):
    import hashlib
    import json
    import pickle
    grid, nml, carry, identity = _myj()
    _write_read(tmp_path, format, grid, nml, carry, identity)
    if format == 'netcdf':
        from gpuwrf.io.netcdf_lock import Dataset
        from gpuwrf.io.wrfrst_netcdf import _optional_variable_name, read_wrfrst_carry
        index = sum(len(jax.tree.leaves(value)) for value in
                    carry.history_diagnostics[:carry.history_diagnostics._fields.index(field)])
        name = _optional_variable_name('history_diagnostics', f'leaf_{index:03d}')
        with Dataset(tmp_path/'carry.nc', 'r+') as ds:
            if damage == 'dtype':
                original = ds[name]
                dims, attrs, values = original.dimensions, {key: original.getncattr(key) for key in original.ncattrs()}, original[...]
                ds.renameVariable(name, name+'_old')
                replacement = ds.createVariable(name, 'f8', dims)
                replacement.setncatts(attrs)
                replacement[...] = values
            elif damage == 'missing':
                ds.renameVariable(name, name+'_old')
            else:
                ds[name][...] = np.nan
        # x64-OFF must not silently repair a corrupt f64 stored wind.
        with jax.enable_x64(False), pytest.raises(ValueError, match='history.'+field if damage != 'missing' else 'missing'):
            read_wrfrst_carry(tmp_path/'carry.nc', namelist=nml)
        return
    if format == 'native':
        from gpuwrf.runtime.restart_store import RestartStore
        path = next((tmp_path/'native').glob('generation-*'))
        payload_path = path/'snapshot.pkl'
        payload = pickle.loads(payload_path.read_bytes())
        extra = payload['carries']['d01']['extra_fields']
    else:
        from gpuwrf.io.restart import read_restart
        payload_path = tmp_path/'carry.pkl'
        payload = pickle.loads(payload_path.read_bytes())
        extra = payload['carry']['extra_fields']
    original = getattr(extra['history_diagnostics'], field)
    bad = (None if damage == 'missing' else np.asarray(original, np.float64)
           if damage == 'dtype' else np.full(np.shape(original), np.nan, np.float32))
    extra['history_diagnostics'] = extra['history_diagnostics']._replace(**{field: bad})
    blob = pickle.dumps(payload)
    payload_path.write_bytes(blob)
    if format == 'native':
        manifest_path = path/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(payload_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
        message = 'restart contains a nonfinite carry leaf' if damage == 'nonfinite' else 'history.'+field
        with pytest.raises(ValueError, match=message):
            RestartStore(tmp_path/'native', 10_000_000, 2, 0).read(path, expected_identity=identity, device=False)
    else:
        with pytest.raises(ValueError, match='history.'+field):
            read_restart(payload_path)


@pytest.mark.parametrize('field', ['t2','q2','u10','v10','lh','psfc'])
def test_ssib_inactive_winds_do_not_skip_other_history_checks(cpu_pallas_interpret, tmp_path, field):
    from test_s2_ssib_wiring import _case
    from gpuwrf.io.restart import write_restart
    grid, _, nml, carry = _case(tmp_path)
    carry = carry.replace(history_diagnostics=carry.history_diagnostics._replace(**{field: None}))
    with pytest.raises(ValueError, match='SSiB restart invalid history.'+field):
        write_restart(carry, nml, grid, 2, tmp_path/'bad.pkl')
