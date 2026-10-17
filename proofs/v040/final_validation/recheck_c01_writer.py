"""Re-publish inactive soil fields with the merged writer on paid GPU files.

CPU histories are opened only after publication, for qualification. All other
paid forecast values and the original files remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('case', type=Path)
parser.add_argument('paid', type=Path)
parser.add_argument('view', type=Path)
a = parser.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert not a.view.exists()
assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3
sys.path.insert(0, str(a.source / 'src'))
import jax
import numpy as np
import netCDF4
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io import wrfout_writer as writer

assert jax.devices()[0].platform == 'cpu'
netCDF4.set_chunk_cache(256 << 10, 521, .75)
SOIL = ('DZS', 'ZS', 'TSLB', 'SMOIS', 'SH2O')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


run_receipt = a.paid / 'gpu_run.json'
run = json.loads(run_receipt.read_text())
assert run['rc'] == 0 and run['platform'] == 'gpu'
assert run['original_cpu_receipt_sha256'] == sha(a.case / 'cpu_run.json')
for name, digest in run['actual_input_sha256'].items():
    assert sha(a.case / 'input' / name) == digest
original = Gen2Run(a.case / 'input')
domain = original.grid('d01')
grid = domain.as_grid_spec()
authority = writer.bind_wrfout_domain_authority('d01', domain, grid)
a.view.mkdir(parents=True)
payload = writer.prepare_wrfout_payload({}, grid, {'sf_surface_physics': 0},
    a.view / 'soil_publication.nc', domain='d01', domain_authority=authority,
    run_start='2023-01-15T00:00:00', valid_time='2023-01-15T00:00:00',
    lead_hours=0., variable_subset=SOIL, full_variable_set=True,
    source_global_attrs={'SF_SURFACE_PHYSICS': np.int32(0)})
writer.write_prepared_wrfout(payload, expected_domain='d01',
    expected_domain_authority=authority, variable_subset=SOIL)
with Dataset(a.view / 'soil_publication.nc') as ds:
    assert len(ds.dimensions['soil_layers_stag']) == 5
    published = {name: np.asarray(ds[name][:]) for name in SOIL}
    metadata = {name: {k: ds[name].getncattr(k) for k in ds[name].ncattrs()}
                for name in SOIL}

# Original CPU-WRF is a reference only, never an input to the publication call.
cpu_checks = []
for path in sorted((a.case / 'cpu').glob('wrfout_d01_*')):
    with Dataset(path) as ds:
        for name in SOIL:
            values = np.asarray(ds[name][:])
            assert values.shape == published[name].shape
            assert values.dtype == published[name].dtype
            assert values.tobytes() == published[name].tobytes(), (path, name)
            cpu_checks.append({'file': path.name, 'field': name,
                'sha256': hashlib.sha256(values.tobytes()).hexdigest()})
assert len(cpu_checks) == 125
(a.view / 'gpu').mkdir()
for name in ('cpu', 'input'):
    (a.view / name).symlink_to((a.case / name).resolve(), target_is_directory=True)
for name in ('cpu_run.json', 'CPU_OUTPUTS.json'):
    shutil.copyfile(a.case / name, a.view / name)
frames = []
for old_path in sorted((a.paid / 'gpu').glob('wrfout_d01_*')):
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    raw_sha = sha(old_path)
    new_path = a.view / 'gpu' / old_path.name
    with Dataset(old_path) as old, Dataset(new_path, 'w', format=old.data_model) as new:
        new.setncatts({k: old.getncattr(k) for k in old.ncattrs()})
        for name, dim in old.dimensions.items():
            size = 5 if name == 'soil_layers_stag' else len(dim)
            new.createDimension(name, None if dim.isunlimited() else size)
        unchanged = {}
        for name, variable in old.variables.items():
            attrs = {k: variable.getncattr(k) for k in variable.ncattrs()}
            options = {'fill_value': attrs.pop('_FillValue')} if '_FillValue' in attrs else {}
            target = new.createVariable(name, variable.datatype, variable.dimensions, **options)
            target.setncatts(attrs)
            if name in SOIL:
                target[:] = published[name]
            else:
                assert 'soil_layers_stag' not in variable.dimensions, name
                values = np.asarray(variable[:])
                target[:] = values
                unchanged[name] = hashlib.sha256(values.tobytes()).hexdigest()
    with Dataset(old_path) as old, Dataset(new_path) as new:
        assert set(old.variables) == set(new.variables)
        for name in old.variables:
            if name in SOIL:
                assert np.asarray(new[name][:]).tobytes() == published[name].tobytes()
                for key, value in metadata[name].items():
                    assert str(new[name].getncattr(key)) == str(value), (name, key)
            else:
                assert np.asarray(old[name][:]).tobytes() == np.asarray(new[name][:]).tobytes(), name
    assert sha(old_path) == raw_sha
    frames.append({'file': old_path.name, 'raw_sha256': raw_sha,
        'published_sha256': sha(new_path), 'unchanged_other_variables': unchanged})
assert len(frames) == 25
receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'case': 'c01',
    'scope': 'Merged host writer publication recheck on copies of paid GPU files; no model advance or CPU-history-fed publication',
    'forecast_receipt': str(run_receipt), 'forecast_receipt_sha256': sha(run_receipt),
    'forecast_source': run['source_commit'],
    'writer_source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.source, text=True).strip(),
    'writer_module_sha256': sha(a.source / 'src/gpuwrf/io/wrfout_writer.py'),
    'view': str(a.view), 'frames': frames, 'soil_fields': list(SOIL),
    'original_CPU_checks': cpu_checks, 'unchanged_all_fields_except_inactive_soil': True,
    'reference': 'Original CPU-WRF; 125/125 published soil field-frames BYTE',
    'F2': 'OPEN', 'cache': 'NONE'}
(a.view / 'PUBLICATION_RECHECK.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('c01: 25 copies changed only five inactive soil arrays/soil dimension; 125/125 BYTE original CPU')
