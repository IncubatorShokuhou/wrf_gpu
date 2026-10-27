"""Actual cycling restart serialization and early refusal, both formats."""
from dataclasses import replace
import pickle

import jax
import numpy as np
import pytest

from gpuwrf.io import restart, wrfrst_netcdf
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state


def save(path, fmt, grid, namelist, carry):
    if fmt == 'pickle':
        restart.write_restart(carry, namelist, grid, 0, path)
    else:
        wrfrst_netcdf.write_wrfrst_carry(carry, grid, namelist, path,
            valid_time='2026-05-04_00:00:00', run_start='2026-05-04_00:00:00', step_index=0)


def read(path, fmt, namelist):
    function = restart.read_restart if fmt == 'pickle' else wrfrst_netcdf.read_wrfrst_carry
    return function(path, namelist=namelist)[0]


@pytest.mark.parametrize('fmt', ['pickle', 'netcdf'])
@pytest.mark.parametrize('cycling', [False, True])
def test_serialized_cycling_identity_and_mode_refusal(tmp_path, fmt, cycling):
    grid = make_ideal_grid(nz=4, ny=3, nx=5)
    state = _state(grid)
    namelist = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, cycling=cycling)
    carry = initial_operational_carry(state)
    path = tmp_path / ('restart.' + fmt)
    save(path, fmt, grid, namelist, carry)
    restored = read(path, fmt, namelist)
    for expected, actual in zip(jax.tree.leaves(carry.state), jax.tree.leaves(restored.state), strict=True):
        assert np.asarray(actual).dtype == np.asarray(expected).dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    with pytest.raises(ValueError, match='cycling.*schema'):
        read(path, fmt, replace(namelist, cycling=not cycling))


@pytest.mark.parametrize('fmt', ['pickle', 'netcdf'])
def test_missing_cycling_marker_refused_before_device_restoration(tmp_path, monkeypatch, fmt):
    grid = make_ideal_grid(nz=4, ny=3, nx=5)
    state = _state(grid)
    namelist = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, cycling=True)
    path = tmp_path / ('restart.' + fmt)
    save(path, fmt, grid, namelist, initial_operational_carry(state))
    if fmt == 'pickle':
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        assert payload.pop('cycling_schema_version') == 1
        with path.open('wb') as stream:
            pickle.dump(payload, stream)
    else:
        with Dataset(path, 'r+') as ds:
            assert ds.GPUWRF_CYCLING_SCHEMA_VERSION == 1
            ds.delncattr('GPUWRF_CYCLING_SCHEMA_VERSION')
    def forbidden(*args, **kwargs):
        raise AssertionError('device restoration before cycling admission')
    monkeypatch.setattr(jax.numpy, 'asarray', forbidden)
    with pytest.raises(ValueError, match='cycling.*schema'):
        read(path, fmt, namelist)
