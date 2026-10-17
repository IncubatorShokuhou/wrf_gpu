"""Read-only attribution of the retained P3 original-WRF/GPU trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for gpu in sorted((a.case / 'gpu').glob('wrfout_d01_*')):
        cpu = a.case / 'cpu' / gpu.name
        with Dataset(str(gpu)) as gd, Dataset(str(cpu)) as cd:
            rain = np.asarray(cd.variables['RAINNC'][0])
            ny, nx = rain.shape
            y, x = np.indices(rain.shape)
            ring = np.minimum.reduce((y, x, ny-1-y, nx-1-x))
            row = {'time': gpu.name, 'fields': {}}
            for name in ('RAINNC', 'SNOWNC', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE',
                         'QNICE', 'QNRAIN', 'QIR', 'QIB', 'TH_OLD', 'QV_OLD',
                         'T', 'T2', 'GLW', 'SWDOWN', 'RE_CLOUD', 'RE_ICE'):
                if name not in gd.variables or name not in cd.variables:
                    continue
                g = np.asarray(gd.variables[name][0], dtype=np.float64)
                c = np.asarray(cd.variables[name][0], dtype=np.float64)
                assert np.isfinite(g).all() and np.isfinite(c).all(), name
                d = g-c
                parts = {}
                for label, mask in (('ring0', ring == 0), ('rings1_4', (ring > 0) & (ring < 5)),
                                    ('interior', ring >= 5), ('CPUwet', rain > .01)):
                    sub = d[..., mask]
                    parts[label] = {'rmse': float(np.sqrt(np.mean(sub**2))) if sub.size else None,
                                    'bias': float(sub.mean()) if sub.size else None}
                row['fields'][name] = {'rmse': float(np.sqrt(np.mean(d**2))),
                                      'bias': float(d.mean()), 'GPUmax': float(g.max()),
                                      'CPUmax': float(c.max()), 'GPUmean': float(g.mean()),
                                      'CPUmean': float(c.mean()), 'parts': parts}
                if name == 'RAINNC':
                    pos = np.unravel_index(np.argmax(np.abs(d)), d.shape)
                    row['rain_worst'] = {'yx': list(map(int, pos)), 'ring': int(ring[pos]),
                                         'GPU': float(g[pos]), 'CPU': float(c[pos])}
                    flat = np.sort((d**2).ravel())
                    row['rain_top20_sse_fraction'] = float(flat[-20:].sum()/max(flat.sum(), 1e-100))
                    row['rain_pattern_corr'] = float(np.corrcoef(g.ravel(), c.ravel())[0, 1]) if np.std(g)*np.std(c)>0 else None
            rows.append(row)
    a.out.write_text(json.dumps({'scope': 'original CPU-WRF vs retained c21 GPU, no new forecast',
                                  'frames': rows}, indent=2, allow_nan=False)+'\n')
    for i, row in enumerate(rows):
        f = row['fields']
        print(i, {k: {m: round(v, 7) for m, v in f[k].items() if m in ('rmse', 'bias', 'GPUmean', 'CPUmean')}
                  for k in ('RAINNC', 'SNOWNC', 'T', 'QVAPOR', 'QICE') if k in f}, flush=True)


if __name__ == '__main__':
    main()
