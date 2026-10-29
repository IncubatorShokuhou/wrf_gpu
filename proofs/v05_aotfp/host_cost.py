"""Bound the additional host-only fingerprint comparison cost (no GPU)."""
import ast
import json
import os
import statistics
import subprocess
import timeit
from pathlib import Path

assert os.environ['JAX_PLATFORMS'] == 'cpu'
from gpuwrf.runtime import aot_executable as ax
source = subprocess.check_output(['git', '-C', os.environ['V05_AOTFP_SOURCE'], 'show',
                                 '286e2442a9:src/gpuwrf/runtime/aot_executable.py'])
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint_matches')
namespace = {}
exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
old = namespace['fingerprint_matches']
live = ax.target_fingerprint()
assert ax.fingerprint_matches(live, live) and old(live, live)
number = 20000
samples = {'base': [], 'candidate': []}
for i in range(7):
    for name in (('base', 'candidate') if i % 2 == 0 else ('candidate', 'base')):
        fn = old if name == 'base' else ax.fingerprint_matches
        samples[name].append(timeit.timeit(lambda: fn(live, live), number=number) / number)
med = {name: statistics.median(vals) for name, vals in samples.items()}
print(json.dumps(dict(samples_s_per_call=samples, median_s_per_call=med,
    delta_s_per_call=med['candidate']-med['base'], scope='CPU predicate only; no GPU/forecast timing claim',
    device_arrays_added=0, device_kernels_added=0, timestep_transfers_added=0,
    memory='eight host field checks and keyword forwarding; no blob copies/extra device buffers'), indent=2))
