"""Acquire active per-field CPU-WRF crops; these are IO samples, not a model state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

p = argparse.ArgumentParser()
p.add_argument('--tree', type=Path, required=True)
p.add_argument('--references', type=Path, required=True)
p.add_argument('--census', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
report = json.loads(a.census.read_text())
with np.load(a.tree / 'src/gpuwrf/physics/p3/data/2mom.npz') as original_tables:
    diameter_max = float(original_tables['ice'][..., 10].max())
    density_max = float(original_tables['ice'][..., 11].max())
table = a.tree / 'src/gpuwrf/physics/p3/data/2mom.npz'
report['bounds'] = dict(p3_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
    p3_diameter_max_m=diameter_max, p3_density_max_kg_m3=density_max,
    jensen_density_max_kg_m3=920., source='P3 original table channels 11/12; Jensen shape.py WRF rho_ice=920',
    other_fields='finite and nonnegative; no invented universal upper bound',
    rounding_relative_margin=32*float(np.finfo(np.float32).eps))
samples = {}
for case, row in report['cases'].items():
    pending = set(row['fields'])
    for name, field in row['fields'].items():
        assert field['min'] >= 0., (case, name, field['min'])
        upper = (920. if case == 'c87' else density_max) if name.startswith('RHO_ICE') else (
            diameter_max if case != 'c87' and name.startswith('D_ICE') else None)
        if upper is not None:
            assert field['max'] <= upper*(1+32*np.finfo(np.float32).eps), (case, name, field['max'], upper)
    for path in sorted((a.references / case / 'cpu').glob('wrfout_d01_*'))[1:]:
        with Dataset(path) as ds:
            for name in sorted(pending):
                values = np.asarray(ds[name][0])
                if np.count_nonzero(values) == 0:
                    continue
                point = np.asarray(np.unravel_index(np.argmax(values), values.shape))
                size = np.asarray((3,2,2))
                origin = np.minimum(np.maximum(point-np.asarray((1,0,0)),0),np.asarray(values.shape)-size)
                index = tuple(slice(int(lo),int(lo+n)) for lo,n in zip(origin,size))
                sample = values[index].copy()
                assert sample.shape == (3,2,2) and sample.dtype == np.float32 and np.isfinite(sample).all()
                assert np.count_nonzero(sample) > 0
                samples[case+'__'+name] = sample
                row['fields'][name]['sample'] = dict(file=path.name,origin_zyx=origin.tolist(),
                    shape=list(sample.shape),sha256=hashlib.sha256(sample.tobytes()).hexdigest())
                pending.remove(name)
        if not pending:
            break
    assert not pending, (case, pending)
a.out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out / 'cpu_samples.npz', **samples)
report['sample_scope'] = 'independent active field crops at recorded locations; IO transcription only, not a coherent model state'
report['sample_sha256'] = hashlib.sha256((a.out/'cpu_samples.npz').read_bytes()).hexdigest()
(a.out / 'cpu_census.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
