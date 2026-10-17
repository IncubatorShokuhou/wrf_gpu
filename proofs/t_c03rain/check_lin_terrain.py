"""Compare Lin's zsfc=0 and pristine caller's HT on exactly the same columns."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.microphysics_lin import _lin_columns


def parse(path):
    scalars, columns = {}, {}
    for line in path.read_text().splitlines():
        match = re.match(r'^([A-Z0-9_]+)\[(\d+)\]=\s*(\S+)', line)
        if match:
            columns.setdefault(match[1], []).append(float(match[3]))
        elif '=' in line:
            key, value = line.split('=', 1)
            scalars[key] = float(value)
    return scalars, columns


def check(root):
    paths = sorted(root.glob('case_*_ht*.txt'))
    parsed = [parse(p) for p in paths]
    args = [jnp.asarray([d[1][name] for d in parsed], jnp.float64) for name in
            ('TH_IN', 'QV_IN', 'QC_IN', 'QR_IN', 'QI_IN', 'QS_IN', 'QG_IN', 'RHO', 'PII', 'P', 'Z', 'DZ8W')]
    ht = jnp.asarray([d[0]['HT'] for d in parsed], jnp.float64)
    result = []
    for label, terrain in [('old_zsfc0', jnp.zeros_like(ht)), ('pristine_HT', ht)]:
        out = {k: np.asarray(v) for k, v in _lin_columns(*args, terrain, 90.0).items()}
        out['rainncv'] = out['pptrain'] + out['pptsnow'] + out['pptgraul'] + out['pptice']
        out['snowncv'] = out['pptsnow'] + out['pptice']
        out['graupelncv'] = out['pptgraul']
        for i, (path, (s, c)) in enumerate(zip(paths, parsed)):
            row = {'column': path.name, 'route': label, 'fields': {}}
            for key, name in [('th','TH_OUT'), ('qv','QV_OUT'), ('qc','QC_OUT'), ('qr','QR_OUT'),
                              ('qi','QI_OUT'), ('qs','QS_OUT'), ('qg','QG_OUT'),
                              ('rainncv','RAINNCV'), ('snowncv','SNOWNCV'), ('graupelncv','GRAUPELNCV')]:
                ref = np.asarray(c[name] if name in c else s[name])
                actual = out[key][i]
                assert np.isfinite(ref).all() and np.isfinite(actual).all()
                error = float(np.max(np.abs(actual-ref)))
                if key == 'th': limit = 1e-2
                elif key in ('rainncv','snowncv','graupelncv'): limit = max(5e-4, 0.015*float(np.max(np.abs(ref))))
                else: limit = max(1e-7, 0.01*float(np.max(np.abs(ref))))
                row['fields'][key] = {'max_abs': error, 'limit': limit, 'pass': error <= limit}
            row['pass'] = all(r['pass'] for r in row['fields'].values())
            result.append(row)
    output = {'reference': 'original pristine Lin REAL4 public entry', 'rows': result}
    (root / 'comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    for row in result:
        print(row['column'], row['route'], row['pass'], 'rainerr', row['fields']['rainncv']['max_abs'],
              'failed', [k for k,v in row['fields'].items() if not v['pass']])


if __name__ == '__main__':
    check(Path(sys.argv[1]))
