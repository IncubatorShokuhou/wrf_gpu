"""Independent active original CPU-WRF values and metadata through each writer mode."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case, writer_authority

DATA = Path(__file__).resolve().parents[1] / 'proofs/t_icediag'


@pytest.mark.parametrize('case,mp', [('c21',50), ('c23',52), ('c87',55)])
@pytest.mark.parametrize('mode', ['full', 'operational', 'subset'])
def test_original_cpu_field_values_metadata_and_bounds(case, mp, mode, tmp_path):
    manifest = json.loads((DATA/'cpu_census.json').read_text())
    path = DATA/'cpu_samples.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sample_sha256']
    row = manifest['cases'][case]; assert row['frames'] == 25
    with np.load(path) as data:
        expected = {name:data[case+'__'+name].copy() for name in row['fields']}
    state, grid, nl = synthetic_case(nx=2,ny=2,nz=3); nl['mp_physics'] = mp
    for name, value in expected.items():
        field = row['fields'][name]
        assert field['min'] >= 0 and field['nonzero'] > 0
        assert np.count_nonzero(value) > 0 and np.isfinite(value).all()
        assert hashlib.sha256(value.tobytes()).hexdigest() == field['sample']['sha256']
        if name.startswith('RHO_ICE'):
            cap = 920. if mp == 55 else manifest['bounds']['p3_density_max_kg_m3']
            assert field['max'] <= cap*(1+manifest['bounds']['rounding_relative_margin'])
        if name.startswith('D_ICE') and mp != 55:
            assert field['max'] <= manifest['bounds']['p3_diameter_max_m']*(1+manifest['bounds']['rounding_relative_margin'])
        setattr(state, name.lower(), value)
    subset = tuple(expected) if mode == 'subset' else None
    now = datetime(2023,1,15,tzinfo=timezone.utc); authority = writer_authority(grid,'d01')
    payload = prepare_wrfout_payload(state,grid,nl,tmp_path/'history.nc',domain='d01',
        domain_authority=authority,valid_time=now,run_start=now,lead_hours=0,
        full_variable_set=mode != 'operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    with Dataset(payload.target) as ds:
        for name, value in expected.items():
            var = ds[name]; schema = row['fields'][name]['schema']
            assert var.dimensions == tuple(schema['dimensions']) and str(var.dtype) == schema['dtype']
            assert np.asarray(var[0]).tobytes() == value.tobytes(), name
            for key, wanted in schema['attrs'].items():
                assert var.getncattr(key) == wanted, (case,name,key)
