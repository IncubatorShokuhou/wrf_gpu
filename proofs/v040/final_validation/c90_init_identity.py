"""Input-only tracer correction: compare both resolved operational initializations.

CPU portable facade is used for both loaders. This proves sameness of inputs
to the unchanged GPU program, not CPU-facade numerical fidelity to WRF.
"""
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

source, output = map(Path, sys.argv[1:3])
assert os.environ['JAX_PLATFORMS'] == 'cpu' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
sys.path[:0] = [str(source / 'src'), str(source / 'proofs/o1_nlbind')]
import gpuwrf
import jax
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as sc
sc._gpu_device = lambda: jax.devices('cpu')[0]
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.gen2_accessor import Gen2Run

def canonical(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        a = np.asarray(value)
        return dict(array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                    shape=a.shape, dtype=str(a.dtype), weak_type=bool(getattr(value, 'weak_type', False)),
                    committed=bool(getattr(value, 'committed', False)))
    if is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(type(value), '__slots__'):
        slots = []
        for cls in type(value).__mro__:
            own = cls.__dict__.get('__slots__', ())
            slots.extend((own,) if isinstance(own, str) else own)
        slots = [s for s in slots if s not in ('__dict__', '__weakref__')]
        return dict(type=type(value).__module__ + '.' + type(value).__qualname__,
                    slot_order=slots, attrs={s: canonical(getattr(value, s, None)) for s in slots})
    if hasattr(value, '__dict__'):
        return dict(type=type(value).__module__ + '.' + type(value).__qualname__, attrs=canonical(vars(value)))
    raise TypeError(f'Unexamined initialization object {type(value)!r}')

root = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
rows = []
started = time.monotonic()
output.mkdir(parents=True, exist_ok=False)
for ident in ('c46', 'c90'):
    case = root / ident
    config = pipeline.NestedPipelineConfig(case / 'input', output / ident / 'output',
        output / ident / 'proof', hours=24, max_dom=1, scratch_dir=output / ident / 'scratch')
    hierarchy, bundles, meta, run_start, dts, carries = pipeline._load_domains(config, ('d01',))
    run = Gen2Run(case / 'input')
    from gpuwrf.timeseries.runtime import prepare_time_series
    _, carries = prepare_time_series(config, ('d01',), bundles, carries, dts, run_start, run.namelist)
    from gpuwrf.streams.runtime import prepare_streams
    _, carries = prepare_streams(config, ('d01',), carries, dts, run_start,
        resumed=False, restart_store=None, restart_identity=None, saved=None)
    resolved = canonical(dict(hierarchy=hierarchy, bundles=bundles, carries=carries,
        run_start=run_start, dt=dts))
    (output / f'{ident}-resolved.json').write_text(json.dumps(resolved, indent=2, allow_nan=False) + '\n')
    rows.append(resolved)
    print(ident, 'resolved', time.monotonic()-started, flush=True)
    del hierarchy, bundles, carries, run, meta, resolved
    jax.clear_caches()
    gc.collect()
def differences(a, b, path=''):
    if type(a) != type(b):
        return [path + ':type']
    if isinstance(a, dict):
        result = [path + ':keys'] if a.keys() != b.keys() else []
        for key in a.keys() & b.keys(): result += differences(a[key], b[key], path + '/' + key)
        return result
    if isinstance(a, list):
        if len(a) != len(b): return [path + ':length']
        return sum((differences(x, y, path + '/' + str(i)) for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [path]
diff = differences(*rows)
receipt = dict(utc=datetime.now(timezone.utc).isoformat(),
    source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
    case_ids=['c46', 'c90'], resolved_differences=diff, resolved_identical=not diff,
    wall_s=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    physical_backend=jax.devices()[0].platform, GPU_executed=False,
    scope='Both full initial carries, bundle/state/namelist/grid/metrics, clock and time-series/streams seeds. '
          'Same-source CPU-facade identity only; original-WRF active-tracer trajectory qualification is separate.')
(output / 'IDENTITY.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
assert not diff, diff[:20]
