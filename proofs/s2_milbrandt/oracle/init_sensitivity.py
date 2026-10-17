"""Compare untouched pristine bodies under two compiler-defined initializations.

No candidate equations or JAX execution are involved. Run on lane CPU cores;
BUILD is disposable, OUT is the retained JSON receipt.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys
import numpy as np

build, fixture, out = map(Path, sys.argv[1:])
build.mkdir(parents=True, exist_ok=True)
source = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_milbrandt2mom.F')
body = source.read_text().split('END MODULE my2_mod', 1)[0] + 'END MODULE my2_mod\n'
(build/'my2.f90').write_text(body)
with np.load(fixture) as data:
    a, ps, dt = data['inputs'], data['ps'], data['dt']
(build/'input.bin').write_bytes(struct.pack('iif', *a.shape[:2], dt) + a.tobytes(order='F') + ps.tobytes(order='F'))
fc = os.environ.get('FC', '<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran')
flags = ['-O2', '-ftree-vectorize', '-funroll-loops', '-ffree-form', '-ffree-line-length-none', '-fcheck=bounds']
results = {}
for label, init in [('zero', ['-finit-real=zero', '-finit-integer=0', '-finit-logical=false']),
                    ('snan', ['-finit-real=snan', '-finit-integer=42', '-finit-logical=true'])]:
    subprocess.run([fc, *flags, *init, '-c', 'my2.f90'], cwd=build, check=True)
    subprocess.run([fc, *flags, str(Path(__file__).with_name('oracle.F90')), 'my2.o', '-o', 'oracle'], cwd=build, check=True)
    subprocess.run([str(build/'oracle'), str(build/'input.bin'), str(build/f'{label}.bin'), 'full'], check=True)
    results[label] = np.fromfile(build/f'{label}.bin', np.float32)
zero, alternate = results['zero'], results['snan']
receipt = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'compiler': fc,
           'flags': flags, 'work': '12 active cold/hail columns x12 levels, DT18; full main boundary',
           'initializations': ['REAL zero / INTEGER 0 / LOGICAL false', 'REAL snan / INTEGER 42 / LOGICAL true'],
           'values': int(zero.size), 'byte_equal': zero.tobytes() == alternate.tobytes(),
           'different_values': int(np.count_nonzero(zero != alternate)),
           'nonfinite_values': {k: int(np.count_nonzero(~np.isfinite(v))) for k, v in results.items()},
           'max_abs': float(np.nanmax(np.abs(zero-alternate)))}
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
assert receipt['byte_equal'], receipt
