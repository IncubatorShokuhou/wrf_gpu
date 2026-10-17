"""Read-only census of the original CPU-WRF c21/c23/c87 diagnostic fields."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = {'scope': 'original CPU-WRF schema and values; not a GPU trajectory gate', 'cases': {}}
    samples = {}
    for case, cats, variables in (('c21', 1, ('D_ICE', 'RHO_ICE', 'V_ICE')),
                                  ('c23', 2, ('D_ICE', 'RHO_ICE', 'V_ICE')),
                                  ('c87', 3, ('D_ICE', 'RHO_ICE', 'V_ICE', 'PHI_ICE'))):
        paths = sorted((a.references / case / 'cpu').glob('wrfout_d01_*'))
        assert len(paths) == 25, (case, len(paths))
        row = result['cases'][case] = {'frames': len(paths), 'fields': {}}
        for path in paths:
            with Dataset(path) as ds:
                for cat in range(1, cats + 1):
                    for base in variables:
                        name = base + (str(cat) if cat > 1 else '')
                        var = ds[name]; x = np.asarray(var[0])
                        assert x.dtype == np.float32 and np.isfinite(x).all(), (path, name)
                        assert var.dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')
                        schema = dict(dimensions=var.dimensions, dtype=str(var.dtype),
                                      attrs={key: var.getncattr(key) for key in
                                             ('description', 'units', 'MemoryOrder', 'FieldType', 'stagger')})
                        schema['attrs'] = {key: value.item() if isinstance(value, np.generic) else value
                                           for key, value in schema['attrs'].items()}
                        f = row['fields'].setdefault(name, dict(schema=schema, min=float(x.min()),
                           max=float(x.max()), nonzero=0, frame_hashes={}))
                        assert f['schema'] == schema
                        f['min'] = min(f['min'], float(x.min())); f['max'] = max(f['max'], float(x.max()))
                        f['nonzero'] += int(np.count_nonzero(x))
                        f['frame_hashes'][path.name] = hashlib.sha256(x.tobytes()).hexdigest()
                        # First evolved frame crop exercises the independent writer transcription.
                        if path == paths[1]:
                            samples[case + '__' + name] = x[:3, :2, :2].copy()
        assert any(f['nonzero'] for f in row['fields'].values()), case
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'cpu_samples.npz', **samples)
    result['sample_sha256'] = hashlib.sha256((a.out / 'cpu_samples.npz').read_bytes()).hexdigest()
    (a.out / 'cpu_census.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
