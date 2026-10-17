"""Apply only the merged CLAT publication result to copies of paid GPU files.

The original forecast and its raw failing score remain untouched. No CPU-WRF
history value is an input to publication; CPU histories are scoring references.
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

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('case', type=Path)
p.add_argument('view', type=Path)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert not a.view.exists()
assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3
sys.path.insert(0, str(a.source / 'src'))
import gpuwrf
import jax
import netCDF4
import numpy as np
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io import wrfout_writer as writer
from gpuwrf.integration.daily_pipeline import _load_static_latlon_writer_diagnostics
from gpuwrf.coupling.boundary_feedback import build_state_feedback_weights
from gpuwrf.nesting.initial_feedback import _input_point_overrides

assert jax.devices()[0].platform == 'cpu'
netCDF4.set_chunk_cache(256 << 10, 521, .75)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(variable):
    values = np.asarray(variable[:])
    return dict(shape=list(values.shape), dtype=str(values.dtype),
                sha256=hashlib.sha256(values.tobytes()).hexdigest())


run_receipt = a.case / 'gpu_run.json'
assert json.loads(run_receipt.read_text())['rc'] == 0
prior = json.loads((a.source / 'proofs/t_writer_clat/C42_CLAT_PRODUCER.json').read_text())
for path in ('src/gpuwrf/nesting/initial_feedback.py',
             'src/gpuwrf/coupling/boundary_feedback.py',
             'src/gpuwrf/coupling/feedback_registry.py'):
    assert subprocess.check_output(['git', 'show', prior['source_commit'] + ':' + path],
                                   cwd=a.source) == (a.source / path).read_bytes()
run = Gen2Run(a.case / 'input')
parent, child = run.grid('d01'), run.grid('d02')
weights = build_state_feedback_weights(parent_grid_ratio=child.parent_grid_ratio,
    i_parent_start=child.i_parent_start, j_parent_start=child.j_parent_start,
    parent_grid=parent, child_grid=child, spec_zone=1, smooth_option=2)
overrides = _input_point_overrides(run, 'd01', 'd02', weights)
producer = np.asarray(overrides['CLAT'])
assert hashlib.sha256(producer.tobytes()).hexdigest() == prior['producer_CLAT_sha256']
grid = parent.as_grid_spec()
diagnostics, _ = _load_static_latlon_writer_diagnostics(run, 'd01', grid=grid)
diagnostics.update(overrides)
authority = writer.bind_wrfout_domain_authority('d01', parent, grid)
a.view.mkdir(parents=True)
payload = writer.prepare_wrfout_payload({}, grid, None, a.view / 'CLAT_publication.nc',
    domain='d01', domain_authority=authority, run_start=datetime(2026, 2, 28),
    valid_time=datetime(2026, 2, 28), lead_hours=0, diagnostics=diagnostics,
    variable_subset=('CLAT', 'XLAT', 'XLONG'), full_variable_set=True)
writer.write_prepared_wrfout(payload, expected_domain='d01',
    expected_domain_authority=authority, variable_subset=('CLAT', 'XLAT', 'XLONG'))
with Dataset(a.view / 'CLAT_publication.nc') as ds:
    published = np.asarray(ds['CLAT'][0])
assert published.dtype == producer.dtype == np.float32
assert published.tobytes() == producer.tobytes()
(a.view / 'gpu').mkdir()
(a.view / 'input').symlink_to((a.case / 'input').resolve(), target_is_directory=True)
(a.view / 'cpu').symlink_to((a.case / 'cpu').resolve(), target_is_directory=True)
for name in ('cpu_run.json', 'CPU_OUTPUTS.json'):
    shutil.copyfile(a.case / name, a.view / name)
frames = []
for original in sorted((a.case / 'gpu').glob('wrfout*')):
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    target = a.view / 'gpu' / original.name
    old_sha = sha(original)
    if original.name.startswith('wrfout_d02_'):
        target.symlink_to(original.resolve())
        frames.append(dict(file=original.name, raw_sha256=old_sha,
                           published_sha256=old_sha, unchanged_child=True))
        continue
    assert original.name.startswith('wrfout_d01_')
    shutil.copyfile(original, target)
    with Dataset(target, 'r+') as ds:
        ds['CLAT'][0] = published
    with Dataset(original) as old, Dataset(target) as new:
        assert set(old.variables) == set(new.variables)
        assert {k: str(old.getncattr(k)) for k in old.ncattrs()} == {
            k: str(new.getncattr(k)) for k in new.ncattrs()}
        unchanged = {}
        for name in old.variables:
            before, after = array_sha(old[name]), array_sha(new[name])
            if name == 'CLAT':
                assert np.asarray(new[name][0]).tobytes() == published.tobytes()
            else:
                assert before == after, (original.name, name)
                unchanged[name] = before
        assert set(old['CLAT'].ncattrs()) == set(new['CLAT'].ncattrs())
    assert sha(original) == old_sha
    frames.append(dict(file=original.name, raw_sha256=old_sha,
        published_sha256=sha(target), unchanged_other_variables=unchanged))
assert len(frames) == 50
receipt = dict(utc=datetime.now(timezone.utc).isoformat(), case='c42',
    scope='Host-only merged CLAT publication recheck on copies of paid full24h GPU files. Original forecast/rawCLAT failures untouched; no model advance or CPU history supplied to publication.',
    forecast_receipt=str(run_receipt), forecast_receipt_sha256=sha(run_receipt),
    forecast_source=json.loads(run_receipt.read_text())['source_commit'],
    writer_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.source, text=True).strip(),
    writer_module_sha256=sha(a.source / 'src/gpuwrf/io/wrfout_writer.py'),
    producer_sha256=hashlib.sha256(producer.tobytes()).hexdigest(),
    producer_sources_BYTE=True, view=str(a.view), frames=frames,
    unchanged_all_fields_except_parent_CLAT=True,
    reference='Preserved original CPU-WRF', F2='OPEN', cache='NONE')
(a.view / 'PUBLICATION_RECHECK.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('c42: 25 parent copies changed onlyCLAT from genuine producer; 25 child symlinks unchanged')
