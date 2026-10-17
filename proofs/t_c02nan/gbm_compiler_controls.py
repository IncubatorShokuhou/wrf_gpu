"""Independent original-WRF controls for the fixed actual c08 columns.

The earlier O0 input-ULP gate remains unchanged; this reports a separate
compiler/input envelope. It never changes a model output or a tolerance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ARRAYS = ('u', 'v', 't', 'qv', 'qc', 'p', 'pii', 'dz', 'tke')
SCALARS = ('psfc', 'znt', 'ust', 'hfx', 'qfx', 'tsk', 'gz1oz0',
           'wspd', 'br', 'psim', 'psih', 'xland', 'dt')


def run(build, values):
    (build / 'column.in').write_text('\n'.join(
        ' '.join(format(float(x), '.17g') for x in np.asarray(values[k]).reshape(-1))
        for k in ARRAYS + SCALARS) + '\n')
    result = subprocess.run([str(build / 'gbm_oracle')], cwd=build,
                            capture_output=True, text=True, check=True, timeout=5)
    out = {}
    for line in result.stdout.splitlines():
        label, value = line.split('=')
        match = re.fullmatch(r'(\w+)\[(\d+)\]', label)
        if match:
            out.setdefault(match[1], []).append(float(value))
        else:
            out[label.strip()] = float(value)
    out = {k: np.asarray(v) for k, v in out.items()}
    assert all(np.isfinite(v).all() for v in out.values()), result.stdout
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plan', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    root = a.plan.parent
    values = np.load(plan['inputs'])
    outputs = np.load(plan['outputs'])
    assert hashlib.sha256(Path(plan['inputs']).read_bytes()).hexdigest() == plan['input_sha256']
    previous = json.loads(Path(plan['prior_gate']).read_text())
    previous = {row['index']: row['fields'] for row in previous['rows']}
    rows = []
    for index in plan['indices']:
        column = {k: np.asarray(values[k] if values[k].ndim == 0 else values[k][index],
                                dtype=np.float32) for k in ARRAYS + SCALARS}
        original = json.loads((Path(plan['baseline']) / f'original_{index}.json').read_text())['output']
        original = {k: np.asarray(v) for k, v in original.items()}
        snan = run(root / 'snan', column)
        init_equal = all(np.array_equal(snan[k], v) for k, v in original.items())
        # Original PBLHGT reads uninitialized kpbl2dx (WRF lines 1020-1050
        # explicitly call it unused). Its integer output is not an oracle.
        defined_equal = all(np.array_equal(snan[k], v) for k, v in original.items() if k != 'kpbl')
        assert defined_equal, 'Defined original outputs depend on scratch initialization'
        envelopes = {}
        for mode in ('optimized-no-fma', 'optimized-fma'):
            build = root / mode
            baseline = run(build, column)
            low = {k: v.copy() for k, v in baseline.items()}
            high = {k: v.copy() for k, v in baseline.items()}
            count = 1
            for key in ARRAYS + SCALARS:
                for position in np.ndindex(column[key].shape):
                    for direction in (-np.inf, np.inf):
                        altered = {k: v.copy() for k, v in column.items()}
                        altered[key][position] = np.nextafter(column[key][position], np.float32(direction))
                        result = run(build, altered)
                        for k, v in result.items():
                            low[k] = np.minimum(low[k], v)
                            high[k] = np.maximum(high[k], v)
                        count += 1
            envelopes[mode] = {'runs': count, 'low': {k:v.tolist() for k,v in low.items()},
                               'high': {k:v.tolist() for k,v in high.items()}}
        fields = {}
        for key in plan['fields']:
            actual = np.asarray(outputs[key][index])
            low = np.asarray(previous[index][key]['low'])
            high = np.asarray(previous[index][key]['high'])
            for env in envelopes.values():
                n = actual.size
                low = np.minimum(low, np.asarray(env['low'][key]).reshape(-1)[:n].reshape(actual.shape))
                high = np.maximum(high, np.asarray(env['high'][key]).reshape(-1)[:n].reshape(actual.shape))
            fields[key] = {'outside': int(np.count_nonzero((actual < low) | (actual > high))),
                           'finite': bool(np.isfinite(actual).all()), 'low': low.tolist(),
                           'high': high.tolist(), 'actual': actual.tolist()}
        rows.append({'index':index, 'initialization_byte_equal':init_equal,
                     'defined_initialization_byte_equal':defined_equal,
                     'undefined_kpbl':{'zero':original['kpbl'].tolist(),'snan':snan['kpbl'].tolist()},
                     'envelopes':envelopes, 'fields':fields})
        print(index, {k:v['outside'] for k,v in fields.items() if v['outside']}, flush=True)
    receipt = {'gate':plan['gate'], 'prior_strict_gate':'FAIL retained', 'rows':rows,
               'verdict':'PASS' if all(v['outside']==0 and v['finite'] for r in rows for v in r['fields'].values()) else 'FAIL'}
    (root/'compiler_gate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['verdict'], flush=True)


if __name__ == '__main__':
    main()
